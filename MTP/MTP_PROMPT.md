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

## Шаг 6. Отладка и валидация

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
