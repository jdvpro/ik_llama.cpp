# Реализация MTP для Qwen3-Next в llama.cpp

## ВАЖНО: Чего НЕ делать

- Это **НЕ Medusa** — никаких параллельных независимых голов
- Это **НЕ отдельная draft модель** — никаких внешних моделей, никакого `-md`
- Это **НЕ Meta-style MTP** с параллельными независимыми модулями
- MTP модули **встроены в основную модель** и работают **последовательно**
- Каждый MTP модуль зависит от выхода предыдущего

---

## Параметры модели

```
num_hidden_layers         = 48   // основные слои: blk.0 .. blk.47
num_nextn_predict_layers  = 1    // MTP слоёв: blk.48
hidden_size               = 2048
hidden_size * 2           = 4096 // размер после конкатенации, вход в eh_proj
```

## Актуальные файлы

Этот проект очень большой. И чтобы не заставлять тебя читать много лишней информации, здесь собраны файлы, которых скорее всего хватит для твоей задачи.
Но конечно, это не 100%. Сначала читай эти файлы, если уже не найдёшь в них то что тебе нужно тогда можешь искать, где тебе кажется оно должно быть.

### Скорее всего пригодятся все

- [common/arg.cpp](common/arg.cpp)
- [common/common.h](common/common.h)
- [common/speculative.cpp](common/speculative.cpp)
- [common/speculative.h](common/speculative.h)
- [include/llama.h](include/llama.h)
- [src/llama-arch.cpp](src/llama-arch.cpp)
- [src/llama-context.cpp](src/llama-context.cpp)
- [src/llama-context.h](src/llama-context.h)
- [src/llama-graph.h](src/llama-graph.h)
- [src/llama-model.cpp](src/llama-model.cpp)
- [src/models/qwen3next.cpp](src/models/qwen3next.cpp)
- [convert_hf_to_gguf.py](convert_hf_to_gguf.py)
- [gguf-py/gguf/constants.py](gguf-py/gguf/constants.py)

### Опционально — может пригодятся, а может нет
- [examples/speculative-simple/speculative-simple.cpp](examples/speculative-simple/speculative-simple.cpp)


---

## Шаг 1. Конвертер. Маппинг тензоров GGUF

Сдесь нужно допилить конвертер, что-бы он в gguf добавлял так же MTP тензоры из safetensor модели. 

MTP блок живёт в `blk.48.*`. Маппинг:

```
blk.48.nextn.hnorm.weight          → pre_fc_norm_hidden       [2048]
blk.48.nextn.enorm.weight          → pre_fc_norm_embedding    [2048]
blk.48.nextn.eh_proj.weight        → fc  [4096, 2048]  (конкатенация → проекция)
blk.48.nextn.shared_head_norm.weight → final_norm             [2048]

blk.48.attn_norm.weight            → decoder.input_layernorm
blk.48.post_attention_norm.weight  → decoder.post_attention_layernorm
blk.48.attn_q.weight               → decoder.attn_q
blk.48.attn_k.weight               → decoder.attn_k
blk.48.attn_v.weight               → decoder.attn_v
blk.48.attn_output.weight          → decoder.attn_output
blk.48.attn_q_norm.weight          → decoder.attn_q_norm
blk.48.attn_k_norm.weight          → decoder.attn_k_norm
blk.48.ffn_gate_inp.weight         → decoder.ffn_gate_inp     (MoE router)
blk.48.ffn_gate_exps.weight        → decoder.ffn_gate_exps
blk.48.ffn_up_exps.weight          → decoder.ffn_up_exps
blk.48.ffn_down_exps.weight        → decoder.ffn_down_exps
blk.48.ffn_gate_shexp.weight       → decoder.ffn_gate_shexp
blk.48.ffn_up_shexp.weight         → decoder.ffn_up_shexp
blk.48.ffn_down_shexp.weight       → decoder.ffn_down_shexp
blk.48.ffn_gate_inp_shexp.weight   → decoder.ffn_gate_inp_shexp
```

Если `num_nextn_predict_layers > 1` — следующие блоки идут как blk.49, blk.50 и т.д.

---

## Шаг 2. Загрузка тензоров

В `llm_load_tensors()` добавить загрузку MTP слоёв в отдельный массив `mtp_layers[]`.

Структура одного MTP слоя:

```cpp
struct llama_mtp_layer {
    // fusion
    struct ggml_tensor * hnorm;              // pre_fc_norm_hidden
    struct ggml_tensor * enorm;              // pre_fc_norm_embedding
    struct ggml_tensor * eh_proj;            // [4096, 2048]
    struct ggml_tensor * shared_head_norm;   // финальная норма перед lm_head

    // декодер-слой (идентичен основным слоям модели)
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * post_attn_norm;
    struct ggml_tensor * attn_q, * attn_k, * attn_v, * attn_out;
    struct ggml_tensor * attn_q_norm, * attn_k_norm;
    struct ggml_tensor * ffn_gate_inp;
    struct ggml_tensor * ffn_gate_exps, * ffn_up_exps, * ffn_down_exps;
    struct ggml_tensor * ffn_gate_shexp, * ffn_up_shexp, * ffn_down_shexp;
    struct ggml_tensor * ffn_gate_inp_shexp;
};
```

**Shared тензоры — не копировать, не создавать новые:**
- `model.tok_embd` — используется для embedding lookup в MTP как есть
- `model.output` (lm_head) — используется для финальных logits MTP как есть

---

## Шаг 3. Forward pass MTP (граф ggml)

Входы:
- `hidden_states` — выход последнего основного слоя (blk.47), shape `[seq_len, 2048]`
- `input_ids_next` — токены сдвинутые на +1 позицию относительно текущих

```
1. embedding = ggml_get_rows(ctx, model.tok_embd, input_ids_next)
   // shape: [seq_len, 2048]

2. embedding = rms_norm(embedding) * enorm          // pre_fc_norm_embedding
   hidden    = rms_norm(hidden_states) * hnorm      // pre_fc_norm_hidden

3. concat = ggml_concat(ctx, embedding, hidden, dim=-1)
   // shape: [seq_len, 4096]

4. fused = ggml_mul_mat(ctx, eh_proj, concat)
   // shape: [seq_len, 2048]

5. fused → стандартный декодер-слой (attention + MoE FFN)
   // архитектура слоя идентична основным слоям blk.0..blk.47
   // RoPE параметры те же что у основной модели
   // positions = pos + 1 (позиции предсказываемых токенов)

6. hidden_out = rms_norm(hidden_after_decoder) * shared_head_norm
   // shape: [seq_len, 2048]

7. logits_mtp = ggml_mul_mat(ctx, model.output, hidden_out)
   // shape: [seq_len, vocab_size]
   // model.output — shared lm_head, тот же что у основной модели
```

Если MTP слоёв несколько: `hidden_out` из шага 6 становится `hidden_states` для следующего MTP блока, `input_ids_next` сдвигается ещё на +1.

---

## Шаг 4. KV Cache

- KV cache для MTP слоёв выделяется **отдельно** от основных слоёв
- Позиции (RoPE): `pos + 1` для первого MTP блока, `pos + 2` для второго и т.д.
- При откате — KV cache MTP откатывается вместе с основным

---

## Шаг 5. Inference loop (режим 1 — без верификации)

```
Итерация N:
  1. Основная модель → logits_main → argmax → token_0
  2. MTP блок 0:
       input: hidden_states из blk.47, token_0 как input_ids_next
       output: logits_mtp_0 → argmax → token_1
  3. Если num_nextn_predict_layers > 1:
       MTP блок 1:
         input: hidden_out из MTP блока 0, token_1 как input_ids_next
         output: logits_mtp_1 → argmax → token_2
  4. Выдаём: [token_0, token_1, ...]
```

Режим 1 — подготовка к режиму 2. Код режима 1 не выкидывается при переходе на режим 2, верификационный цикл добавляется поверх.

---

## Шаг 5.5. Итеративный draft с одним MTP модулем (multi-step speculation)

### Контекст: почему это нужно

У модели `num_nextn_predict_layers = 1` (один MTP модуль). Но это **не означает** ограничение в 1 draft-токен. Один MTP модуль можно прогнать авторегрессивно N раз, каждый раз подавая собственный выход обратно на вход. Именно так работает SGLang с параметрами `--speculative-num-steps 3 --speculative-num-draft-tokens 4`.

Референсные данные по производительности (DeepSeek V3, SGLang):
- 2 draft-токена: средний accept length ~1.9
- 4 draft-токена: средний accept length ~2.44
- Ускорение при batch_size=1: до 2.1x

### Архитектура итеративного draft generation

При `num_nextn_predict_layers = 1` и N_DRAFT_STEPS = 3:

```
Шаг 0 (от main модели):
  h_prev = hidden_states из blk.47  (выход основной модели)
  tok    = token_0                   (выбранный main моделью)
  
  emb     = RMSNorm(Emb(tok)) * enorm
  h_norm  = RMSNorm(h_prev) * hnorm
  concat  = [emb, h_norm]           // dim=-1, shape [1, 4096]
  fused   = eh_proj @ concat        // shape [1, 2048]
  h_mtp_0 = TRM_48(fused)           // transformer block, shape [1, 2048]
  logits  = lm_head(RMSNorm(h_mtp_0) * shared_head_norm)
  D0      = argmax(logits)

Шаг 1 (от MTP выхода):
  h_prev = h_mtp_0                   // ← hidden state ПОСЛЕ transformer block
  tok    = D0                        // ← draft-токен из предыдущего шага

  emb     = RMSNorm(Emb(tok)) * enorm
  h_norm  = RMSNorm(h_prev) * hnorm  // ← тот же hnorm, тот же enorm!
  concat  = [emb, h_norm]
  fused   = eh_proj @ concat
  h_mtp_1 = TRM_48(fused)            // ← тот же transformer block!
  logits  = lm_head(RMSNorm(h_mtp_1) * shared_head_norm)
  D1      = argmax(logits)

Шаг 2:
  h_prev = h_mtp_1
  tok    = D1
  ... (аналогично)
  → D2
```

### КРИТИЧЕСКИ ВАЖНО: что именно передаётся между шагами

**h_prev для шага k+1 — это выход transformer block MTP (`h_mtp_k`), то есть hidden state ПОСЛЕ attention + FFN, но ДО shared_head_norm и ДО lm_head.**

Типичная ошибка: брать состояние после `shared_head_norm` или после `lm_head`. Это неправильно — нормированное/проецированное представление не подходит как вход для следующей итерации.

Диаграмма потока данных:
```
                          ┌─── logits → argmax → D_k (draft token)
                          │
h_prev → [RMSNorm*hnorm] ─┐                  
                           ├─ concat → eh_proj → TRM_48 → h_mtp_k ──┤
Emb(tok) → [RMSNorm*enorm]┘                                         │
                                                                     │
                          ┌──────────────────────────────────────────┘
                          │
                          ▼
                    h_prev для шага k+1
                    (подаётся обратно в RMSNorm*hnorm)
```

### Где именно получить h_mtp_k в коде

В forward pass MTP (Шаг 3 этого документа) есть два отдельных этапа:

1. **Шаг 5**: `fused → декодер-слой → hidden_after_decoder` — это и есть **h_mtp_k**
2. **Шаг 6**: `rms_norm(hidden_after_decoder) * shared_head_norm → hidden_out` — это уже подготовка для lm_head

Для итеративного draft нужен результат шага 5, НЕ шага 6.

### RMSNorm: обязателен на ОБОИХ компонентах

Перед конкатенацией оба компонента должны пройти через RMSNorm:
- `hidden` → `RMSNorm(hidden) * hnorm`  (pre_fc_norm_hidden)
- `embedding` → `RMSNorm(embedding) * enorm`  (pre_fc_norm_embedding)

Без нормализации масштабы hidden state (~десятки) и embedding (~единицы) будут несовместимы, projection matrix выдаст мусор, и acceptance rate на шагах 1+ упадёт до случайного уровня (~15-20%).

### KV Cache для итеративного draft

Каждый итеративный шаг использует **один и тот же** MTP transformer block (blk.48), но с разными позициями:
- Шаг 0: position = pos + 1
- Шаг 1: position = pos + 2
- Шаг 2: position = pos + 3

KV cache MTP блока должен корректно накапливать записи от всех draft шагов. При rejection во время верификации — откатывать KV cache MTP до точки расхождения.

### Диагностика: как понять что итеративный draft работает правильно

Добавить per-position acceptance rate в логи:

```cpp
fprintf(stderr, "[MTP] D0: %d/%d = %.1f%%, D1: %d/%d = %.1f%%, D2: %d/%d = %.1f%%\n",
    acc_d0, total_d0, 100.0f * acc_d0 / total_d0,
    acc_d1, total_d1, 100.0f * acc_d1 / total_d1,
    acc_d2, total_d2, 100.0f * acc_d2 / total_d2);
```

**Ожидаемые значения (temperature=0, greedy):**
- D0: 60-80% (предсказание от hidden state main модели)
- D1: 40-60% (предсказание от hidden state MTP)
- D2: 30-50% (ещё одна итерация, качество деградирует но не катастрофически)

**Красные флаги — реализация сломана:**
- D0 нормальный (60%+), D1 резко падает до <20% → h_prev передаётся неправильно (не тот тензор, нет RMSNorm, нет projection)
- Все позиции <20% → MTP forward pass целиком сломан
- D1 ≈ D0 ≈ D2 — подозрительно, проверить что не подаётся один и тот же hidden state на все шаги

### Параметр N_DRAFT_STEPS

Рекомендуемое значение: **3** (SGLang использует 2-3 по умолчанию для DeepSeek).

Больше 4 обычно не имеет смысла — acceptance rate деградирует с каждым шагом, и overhead от дополнительных MTP forward passes перевешивает выигрыш. На CPU overhead от каждого MTP шага относительно выше чем на GPU (нет CUDA graphs), поэтому оптимум может быть 2-3.

Сделать настраиваемым через параметр командной строки (например `--mtp-draft-steps N`).

---

## Шаг 6. Inference loop (режим 2 — с верификацией / speculative decoding)

### Общая схема

```
Итерация N:
  1. Main модель decode → logits_main → sample → T0
     Сохранить: hidden_states из blk.47

  2. Итеративный MTP draft (N_DRAFT_STEPS раз):
     draft_tokens = [D0, D1, D2]  (см. Шаг 5.5)

  3. Верификация: собрать batch [T0, D0, D1, D2] и прогнать main модель
     - logits[0] проверяет D0: если argmax(logits[0]) == D0 → accept
     - logits[1] проверяет D1: если D0 принят И argmax(logits[1]) == D1 → accept
     - logits[2] проверяет D2: если D0,D1 приняты И argmax(logits[2]) == D2 → accept
     - Первый rejected → стоп, берём token из logits[rejected_pos] как новый T0

  4. Принять все токены до первого rejection
     Откатить KV cache (main + MTP) до точки расхождения

  5. Следующая итерация начинается с нового T0
```

### Backup/Restore KV state

На CPU backup/restore KV-кэша — это memcpy больших буферов, что дорого. Варианты оптимизации:
- Откатывать только seq_len позиций которые были добавлены draft-токенами (не весь кэш)
- Использовать llama_kv_self_seq_rm() для удаления только draft-позиций

### Математика ожидаемого ускорения

При N_DRAFT_STEPS=3 и per-token acceptance rate p:
- Среднее принятых токенов за итерацию: p + p² + p³ + 1 (последний — из verify)
  - p=0.65: 1 + 0.65 + 0.42 + 0.27 = 2.34 токена
  - p=0.70: 1 + 0.70 + 0.49 + 0.34 = 2.53 токена
- Стоимость итерации: 1 main decode + N MTP forwards + 1 verify batch
- На CPU MTP forward ~2-3ms, main decode ~12-15ms, verify batch ~15-20ms
- Breakeven: ускорение начинается когда средний accept length > (cost_iteration / cost_single_decode)

---

## Шаг 7. Отладка и валидация

Ты можешь выбрать один из двух вариантов на свое усмотрение.

### Вариант 1

Добавить логирование в inference loop. Место: сразу после того как MTP выдал token_1 и после того как на следующей итерации основная модель выдала свой token_0.

Сравниваем: MTP предсказал token_1 на итерации N — основная модель выдала token_0 на итерации N+1. Если совпали — acceptance.

Вывод в stderr:

```cpp
fprintf(stderr, "[MTP] predicted: %d, actual: %d, match: %s | rate: %d/%d = %.1f%%\n",
    mtp_token, actual_token,
    mtp_token == actual_token ? "YES" : "NO",
    match_count, total_count,
    100.0f * match_count / total_count);
```

Счётчики обнулять раз в 100 токенов для скользящего среднего.

**Критерий корректной реализации:**
- Acceptance rate 55%+ → архитектура правильная, можно переходить к режиму 2
- Acceptance rate ниже 20% → что-то не так в forward pass, разбираться до режима 2
- Токены осмысленные (ни мусор, ни повторение одного id) → базовая проверка вменяемости

### Вариант 2

Использовать уже существующий скрипт `llama-speculative-simple`, только его предварительно, конечно, надо для наших нужд подправить. Например, убрать требование draft модели.
Плюс этого варианта в том, что он сразу показывает нужны нам параметр — `accept`.
