# Research Plan: Transformer-based Proteomics Modeling

## Plan of Investigation

### **Now – November (T11)**

- Study Transformer architecture in depth (Attention, Encoder/Decoder, Positional Encoding).
- Review and analyze the **classification process for MS/MS** spectra.
- Study existing models: **Prosit** and **AlphaPeptDeep**, focusing on structure, data processing, and training pipeline.

---

### **November – December (T11–T12)**

- Implement baseline models using **PyTorch**:
  - **CCS prediction** → output dimension = 1
  - **RT prediction** → output dimension = 1
  - **MS/MS prediction** → output dimension = 400
- Integrate **RT** and **CCS** pipelines into a unified framework.
- Run **baseline training** on **AutoRT** and **Prosit** datasets.

---

### **January (T1)**

- Optimize **Transformer hyperparameters**:
  - Number of layers
  - Attention heads
  - Dropout rate
  - Embedding dimension (`d_model`)
- Develop a **unified evaluation module** for **RT + CCS + MS/MS** performance comparison.

---

### **February (T2)**

- Conduct **large-scale training** using full datasets.
- Generate and document **intermediate results and analysis report**.

---

### **March (T3)**

- Finalize **model comparison** and **overall analysis**.
- Summarize:
  - Key findings and improvements
  - Limitations of current approaches
  - Future research directions and applications

---

## Expected Outcomes

- Unified Transformer-based pipeline for RT, CCS, and MS/MS prediction.
- Quantitative comparison between Prosit, AlphaPeptDeep, and custom Transformer models.
- Reproducible training and evaluation framework for downstream proteomics research.

---

## Tools & Frameworks

- **Python 3.10+**
- **PyTorch**
- **NumPy, Pandas**
- **Matplotlib / Seaborn** for visualization

---

## References

- Gessulat et al., _Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning_, **Nat Methods (2019)**.
- Wen et al., _AlphaPeptDeep: A modular deep learning framework to predict peptide properties for proteomics_, **Nat Commun (2023)**.
