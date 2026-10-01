Here is a detailed, technical breakdown of the six selected papers. This overview is specifically structured for your **experiment recreation and initial testing phase**, detailing exact probing and steering mechanics, PyTorch/Hugging Face hook targets, mathematical formulations, and how to extract internal transformer states (activations, attention scores, and LM head projections) for predicting and controlling offensiveness.

---

### 1. Multilingual Political Views of LLMs: Identification and Steering
*(Derived from `identification_and_steering.pdf`)*

*   **Goal:** To prove that political and ideological stances in instruction-tuned LLMs are linearly decodable across languages, and to demonstrate that lightweight, inference-time activation steering on specific attention heads can shift the model's stance cross-lingually.
*   **LLMs Tested:** LLaMA-3.1 (8B, 70B), Qwen-3 (8B, 14B, 32B), and Aya-Expanse.
*   **Libraries & Tools:** PyTorch, Hugging Face `transformers` and `datasets` (`JyotiNayak/political_ideologies`), and the Political Compass Test (PCT). Code: `github.com/d-gurgurov/Political-Ideologies-LLMs`.
*   **Internal Extraction & Probing Details:**
    *   *What is extracted:* The value output vectors of **every individual attention head** \\((l, h)\\) across all \\(L\\) layers.
    *   *Probing method:* Train independent binary logistic regression probes on top of head outputs to classify liberal (\\(c=1\\)) vs. conservative (\\(c=0\\)) text.
    *   *Probing findings:* Accuracy peaks at **90%** in mid-to-late transformer layers (specifically Layer 30, Head 19; and Layers 11, 13, 26–31 in LLaMA-3.1-8B).
*   **Steering & Intervention Method:**
    *   *Center-of-Mass (DiffMean) Vector:* For the \\(K\\) most responsive heads (e.g., \\(K=512\\), top 50% of heads), compute mean activations \\(\mu^{(1)}_{l,h}\\) and \\(\mu^{(0)}_{l,h}\\) for opposing classes, normalize, and take the difference:
        \\[\vec{v}_{l,h} = \frac{\mu^{(1)}_{l,h} - \mu^{(0)}_{l,h}}{\|\mu^{(1)}_{l,h} - \mu^{(0)}_{l,h}\|}\\]
    *   *Inference Injection:* Via PyTorch forward hooks on attention head value outputs:
        \\[\text{value}_{l,h} \leftarrow \text{value}_{l,h} + \alpha \cdot \sigma_{l,h} \cdot \vec{v}_{l,h}\\]
        where \\(\sigma_{l,h}\\) is the standard deviation of activation projections and \\(\alpha\\) is intervention strength (\\(\alpha \approx 20\\)).
*   **Utility for Offensiveness Recreation:** PyTorch forward hooks placed directly on attention head value outputs allow you to steer the model toward specific group perspectives without re-training.

---

### 2. FairSteer: Inference-Time Debiasing via Dynamic Activation Steering
*(Derived from `inference_time_debiasing.pdf`)*

*   **Goal:** Mitigate social biases in LLM outputs dynamically during inference without prompt engineering or fine-tuning, preserving general language model performance on downstream tasks.
*   **LLMs Tested:** Llama-2-chat (7B, 13B), Llama-3-instruct (8B), Vicuna-v1.5 (7B, 13B), and Mistral-v0.3-instruct (7B).
*   **Libraries & Tools:** PyTorch, Hugging Face `evaluate`, BBQ (58k samples), MMLU, UNQOVER, CrowS-Pairs, CEB. Code: `github.com/LiYichen99/FairSteer`.
*   **Internal Extraction & Probing Details:**
    *   *What is extracted:* Residual stream activations \\(a_n^l \in \mathbb{R}^d\\) at the **last token position** (\\(t_n\\)) before next-token generation across all layers \\(l\\).
    *   *Biased Activation Detection (BAD):* Train a linear classifier \\(C^l\\) per layer using cross-entropy loss with \\(L_2\\) regularization:
        \\[\hat{y} = \sigma(w^T a^l + b)\\]
    *   *Layer Selection:* Mid-layers (Layers 13 to 15) achieve peak validation accuracy (>90%), making them optimal intervention points.
*   **Steering & Intervention Method:**
    *   *Debiasing Steering Vector (DSV):* Computed from \\(N=110\\) contrastive prompt pairs \\((P^+, P^-)\\) differing only in choice options:
        \\[v^l = \frac{1}{|D_{\text{DSV}}|} \sum_{(P^+, P^-)} [a^l(P^+) - a^l(P^-)]\\]
    *   *Dynamic Activation Steering (DAS):* At inference, if the classifier predicts bias (\\(\hat{y} < 0.5\\)), dynamically patch the residual stream at layer \\(l^*\\):
        \\[a^{\text{adj}}_{l^*} = a_{l^*} + v^{l^*}\\]
*   **Utility for Offensiveness Recreation:** Provides a dynamic "trigger" architecture. You can train a linear classifier on residual stream activations to detect when an input enters a controversial, high-disagreement zone, triggering a group-specific steering vector only when needed.

---

### 3. Steering Towards Fairness: Mitigating Political Bias in LLMs
*(Derived from `mitigating_political_bias.pdf`)*

*   **Goal:** Probe and neutralize political/ideological biases in decoder LLMs across English, Urdu, and Punjabi using multi-layer Steering Vector Ensembles.
*   **LLMs Tested:** Mistral-7B-Instruct-v0.2 and DeepSeek-VL-7B-Chat.
*   **Libraries & Tools:** PyTorch, Hugging Face `transformers`, Multilingual Political Compass Test (PCT) dataset. Code: `github.com/Afx-Msh/SVE_Mitigation`.
*   **Internal Extraction & Probing Details:**
    *   *What is extracted:* Last-token hidden activations \\(h^{(l)}(x) \in \mathbb{R}^d\\) across transformer layers.
    *   *Individual Steering Vectors (ISV):* Train logistic regression classifiers on StandardScaler-normalized activations from positive vs. negative contrastive pairs. Normalize coefficients to unit length: \\(v_l = \theta / \|\theta\|\\).
*   **Steering & Intervention Method:**
    *   *Steering Vector Ensembles (SVE):* Instead of single-layer interventions, aggregate ISVs across mid-level layers \\(l \in \{8, 12, 16, 20, 24\}\\) weighted by a vector-quality score:
        \\[q_l = 0.6 \cdot \text{accuracy}_l + 0.4 \min\left(\frac{\text{separation}_l}{2}, 1.0\right)\\]
        \\[w_l = \frac{q_l}{\sum_i q_i}, \quad v_{\text{SVE}} = \frac{\sum_l w_l v_l}{\|\sum_l w_l v_l\|}\\]
    *   *Inference Injection:* Simultaneous PyTorch forward hook addition on residual streams across layers \\(l \in \{8, 12, 16, 20, 24\}\\):
        \\[h^{(l)}(x)' = h^{(l)}(x) + \alpha \cdot v_{\text{SVE}} \quad (\alpha = 1.0)\\]
*   **Utility for Offensiveness Recreation:** Multi-layer ensemble steering (\\(v_{\text{SVE}}\\)) provides significantly higher stability and preserves text fluency far better than intervening on a single layer.

---

### 4. Probing Political Ideology: Generalization Across Tasks
*(Derived from `probing_political_ideology.pdf`)*

*   **Goal:** Investigate whether latent ideological directions discovered via linear probes on lawmaker DW-NOMINATE scores causally generalize to downstream reasoning tasks (bias detection, voting simulation, text neutralization).
*   **LLMs Tested:** LLaMA-2-7B-chat, LLaMA-3.1-8B-instruct, and Qwen-2.5-7B-instruct.
*   **Libraries & Tools:** PyTorch, Hugging Face `transformers`, `scikit-learn` (Ridge Regression). Code: `github.com/DotIN13/linear-political-llm`.
*   **Internal Extraction & Probing Details:**
    *   *What is extracted:* Activations \\(x_{l,h}^{(i)}\\) of individual attention heads across \\(L\\) layers and \\(H\\) heads.
    *   *Probing method:* Train independent Ridge Regression probes to predict lawmaker DW-NOMINATE continuous scores \\(y^{(i)}\\):
        \\[\hat{y}_{l,h}^{(i)} = \theta_{l,h}^\top x_{l,h}^{(i)}\\]
*   **Steering & Intervention Method:**
    *   *Intervention:* Select the top-\\(k\\) predictive attention heads ranked by probe \\(R^2\\) (\\(k \in \{8, 16, 32, 64, 96\}\\)).
    *   *Steering Formula:* Modify head activations at every generation step:
        \\[x_{l,h} \leftarrow x_{l,h} + \alpha \cdot \sigma_{l,h} \cdot \theta_{l,h}\\]
        where \\(\sigma_{l,h}\\) is the empirical standard deviation of activations at head \\((l,h)\\) and \\(\alpha \in [-30, 30]\\).
*   **Utility for Offensiveness Recreation:** Proves that shifting a model along an internal ideological vector symmetrically flips its perception of bias/offensiveness in text.

---

### 5. PASTA: Post-Hoc Attention Steering Approach
*(Derived from `post_hoc_attention_steering.pdf`)*

*   **Goal:** Force an LLM to attend intensely to user-specified prompt tokens (e.g., target identity groups or instructions) at inference time without modifying model weights or re-training.
*   **LLMs Tested:** LLaMA-7B, LLaMA-13B, GPT-J-6B, and Vicuna-7B-v1.3.
*   **Libraries & Tools:** PyTorch, Hugging Face `transformers`. Code: `github.com/QingruZhang/PASTA`.
*   **Internal Extraction & Profiling Details:**
    *   *What is accessed:* Raw multi-head attention score matrices \\(A^{(l,h)} = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)\\).
    *   *Multi-Task Model Profiling:* Subsample small datasets (\\(|D|=200..1000\\)) across tasks. Evaluate the accuracy impact of steering each individual head \\((l,h)\\) and select the top-\\(k\\) intersection set \\(H = \bigcap_{i=1}^m R^{(i)}_{1:k}\\) (typically 50 to 100 heads).
*   **Steering & Reweighting Mechanism:**
    *   For specified input tokens \\(G\\) (and non-specified tokens \\(G^- = [n] - G\\)), downweight non-specified attention scores by scaling coefficient \\(\alpha \approx 0.01\\) and re-normalize:
        \\[[T(A)]_{ij} = \begin{cases} \frac{\alpha A_{ij}}{C_i} & \text{if } j \in G^- \\ \frac{A_{ij}}{C_i} & \text{otherwise} \end{cases}\\]
        where \\(C_i = \sum_{j \in G} A_{ij} + \sum_{j \in G^-} \alpha A_{ij}\\).
*   **Utility for Offensiveness Recreation:** In offensiveness detection, attention often skips subtle cultural cues or target identity tokens. PASTA lets you programmatically force the model's attention heads to focus on specific cultural or demographic keywords in the prompt.

---

### 6. TDD: Token Distribution Dynamics
*(Derived from `unveiling_and_manipulating_prompt_influence.pdf`)*

*   **Goal:** Estimate input token saliency by projecting internal layer activations into vocabulary space, and manipulate prompts for zero-shot toxic language suppression and sentiment steering.
*   **LLMs Tested:** GPT-2 (Large/XL), GPT-J-6B, BLOOM-7B, Pythia-6.9B, LLaMA2-7B, LLaMA2-13B, and OPT-30B.
*   **Libraries & Tools:** PyTorch, Hugging Face `transformers`, BLiMP, RealToxicityPrompts, Perspective API. Code: `github.com/zijian678/TDD`.
*   **Internal Extraction & Projection Details:**
    *   *The LM Head as an Interpreter:* Pass hidden state vectors \\(x_i^l \in \mathbb{R}^d\\) of **all input tokens** \\(w_i\\) at layer \\(l\\) directly through the pre-trained Language Model head \\(M_h\\) to map them to vocabulary space:
        \\[p_i^l = \text{softmax}(M_h x_i^l) \in \mathbb{R}^{|V|}\\]
*   **Saliency & Control Mechanics:**
    *   *Token Distribution Dynamics (TDD):* Measure token saliency leading the model to generate target token \\(w_t\\) over alternative token \\(w_a\\):
        \\[c_i^{\text{forward}} = p_{i,n}^L(w_t) - p_{i,n}^L(w_a)\\]
        \\[c^{\text{bidirectional}}_i = c^{\text{forward}}_i + c^{\text{backward}}_i\\]
    *   *Zero-Shot Toxic Suppression:* Identify toxic trigger tokens in the input prompt (top 15% saliency) using TDD against WordFilter target words, and replace them with space tokens.
*   **Utility for Offensiveness Recreation:** Allows you to project intermediate activations of *any* input token directly into vocabulary space (`model.lm_head(hidden_state)`) to see what words the model is "thinking" at any layer, giving you word-level interpretability of offensive triggers.

---

### Summary Checklist for Baseline Testing

1.  **Extracting Residual Activations:** Attach PyTorch forward hooks to `model.model.layers[l]` to capture \\(h^{(l)}(x)\\) at the last token position.
2.  **Extracting Attention Head Outputs:** Hook into `model.model.layers[l].self_attn.o_proj` to extract or modify individual head value outputs.
3.  **Inspecting Hidden States via LM Head:** Pass any token's hidden state \\(x_i^l\\) through `model.lm_head` to inspect top vocabulary tokens.
4.  **Steering Vectors:** Compute difference-of-means vectors \\(v = \mu^+ - \mu^-\\) using small contrastive pairs and inject them into mid-level layers (\\(l \in [13..24]\\)) during inference.