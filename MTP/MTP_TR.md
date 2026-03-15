
## Multi-Token Prediction

Inspired by Gloeckle et al. (2024), we investigate and set a Multi-Token Prediction (MTP) objective for DeepSeek-V3, which extends the prediction scope to multiple future tokens at each position. On the one hand, an MTP objective densifies the training signals and may improve data efficiency. On the other hand, MTP may enable the model to pre-plan its representations for better prediction of future tokens. Figure 3 illustrates our implementation of MTP. Different from Gloeckle et al. (2024), which parallelly predicts 𝐷 additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth. We introduce the details of our MTP implementation in this section. MTPModules. Tobespecific,ourMTPimplementationuses 𝐷sequentialmodulestopredict 𝐷 additional tokens. The 𝑘-th MTP module consists of a shared embedding layer Emb(·), a shared output head OutHead(·), a Transformer block TRM𝑘(·), and a projection matrix 𝑀𝑘 ∈ R𝑑×2𝑑. For the 𝑖-th input token 𝑡𝑖, at the 𝑘-th prediction depth, we first combine the representation of the 𝑖-th token at the (𝑘−1)-th depth h𝑘−1 𝑖 ∈ R𝑑 andthe embedding of the (𝑖+𝑘)-th token 𝐸𝑚𝑏(𝑡𝑖+𝑘) ∈ R𝑑 10with the linear projection:

```
h′𝑘 𝑖 = 𝑀𝑘[RMSNorm(h𝑘−1 𝑖 ); RMSNorm(Emb(𝑡𝑖+𝑘))],
```


where [·;·] denotes concatenation. Especially, when 𝑘 = 1, h𝑘−1 𝑖 (21) refers to the representation given by the main model. Note that for each MTP module, its embedding layer is shared with the main model. The combined h′𝑘 𝑖 serves as the input of the Transformer block at the 𝑘-th depth to produce the output representation at the current depth

```
h𝑘 𝑖 : h𝑘 1:𝑇−𝑘 = TRM𝑘(h′𝑘 1:𝑇−𝑘 ),
```

where𝑇 represents the input sequence length and 𝑖:𝑗 denotes the slicing operation (inclusive of both the left and right boundaries). Finally, taking h𝑘 𝑖 as the input, the shared output head will compute the probability distribution for the 𝑘-th additional prediction token 𝑃𝑘 𝑖+1+𝑘 ∈ R𝑉, where 𝑉 is the vocabulary size:

```
𝑃𝑘 𝑖+𝑘+1 = OutHead(h𝑘 𝑖 ).
```

Theoutput headOutHead(·) linearly maps the representation to logits and subsequently applies the Softmax(·) function to compute the prediction probabilities of the 𝑘-th additional token. Also, for each MTP module, its output head is shared with the main model. Our principle of maintaining the causal chain of predictions is similar to that of EAGLE (Li et al., 2024b), but its primary objective is speculative decoding (Leviathan et al., 2023; Xia et al., 2023), whereas we utilize MTP to improve training. 

### MTPTraining Objective.

For each prediction depth, we compute a cross-entropy loss L𝑘 MTP:

```
L𝑘 MTP = CrossEntropy(𝑃𝑘 2+𝑘:𝑇+1,𝑡2+𝑘:𝑇+1) = −1 𝑇 𝑇+1 ∑ ︁ 𝑖=2+𝑘 log 𝑃𝑘 𝑖 [𝑡𝑖],
```

where𝑇 denotes the input sequence length, 𝑡𝑖 denotes the ground-truth token at the 𝑖-th position, and 𝑃𝑘 𝑖 [𝑡𝑖] denotes the corresponding prediction probability of 𝑡𝑖, given by the 𝑘-th MTP module. Finally, we compute the average of the MTP losses across all depths and multiply it by a weighting factor 𝜆 to obtain the overall MTP loss LMTP, which serves as an additional training objective for DeepSeek-V3:

```
LMTP = 𝜆 𝐷 𝐷∑︁ 𝑘=1 L𝑘 MTP.
```

### MTPinInference. 

OurMTPstrategy mainly aims to improve the performance of the main model, so during inference, we can directly discard the MTP modules and the main model can function independently and normally. Additionally, we can also repurpose these MTP modules for speculative decoding to further improve the generation latency.