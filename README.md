<div align="center">

<h1><img src="docs/resources/icons/scan.ico" width="4%" style="margin-right: 10px"/>DeepTumorVQA 1.0</h1>
<h3><i>A Large-Scale Medical Visual Question Answering Benchmark</i></h3>

<img src="images/task_demonstration.png" width="55%" alt="Dataset teaser"/>

<p>
<a href="https://huggingface.co/datasets/tumor-vqa/DeepTumorVQA_1.0" target="_blank">
    <img alt="HF Dataset" src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface" height="20" />
</a>
</p>

<div style="font-size: 0.9em; text-align: center;">

<a href="#" target="_blank">Yixiong Chen</a><sup>1</sup> •
<a href="#" target="_blank">Pedro R. A. S. Bassi</a><sup>1,2</sup> •
<a href="#" target="_blank">Wenjie Xiao</a><sup>1</sup> •
<a href="#" target="_blank">Xinze Zhou</a><sup>1</sup> •
<a href="#" target="_blank">Sezgin Er</a><sup>3</sup> •
<a href="#" target="_blank">Ibrahim Ethem Hamamci</a><sup>3</sup> •
<a href="#" target="_blank">Zongwei Zhou</a><sup>1</sup> •
<a href="#" target="_blank">Alan Yuille</a><sup>1</sup>

</div>

<div style="font-size: 0.9em; text-align: center;">

<sup>1</sup> Johns Hopkins University &emsp;
<sup>2</sup> University of Bologna &emsp;
<sup>3</sup> Istanbul Medipol University  
<br/>
Contact: <a href="mailto:ayuille1@jhu.edu">ayuille1@jhu.edu</a>

</div>


---

## 🧠 Overview
We present **DeepTumorVQA**, a diagnostic visual question answering (VQA) benchmark targeting abdominal tumors in CT scans. It comprises **9,262 CT volumes** (3.7M slices) from 17 public datasets, with **395K** expert-level questions spanning four categories: Recognition, Measurement, Visual Reasoning, and Medical Reasoning.

---

## 📁 Dataset Format

Each example contains the following fields:

- `image`: Path or identifier for the CT image or volume
- `question`: A natural-language question about the image
- `answer`: The corresponding answer (expert-level)
- (Optional: metadata like `modality`, `region`, `split`, etc.)

### 🔍 Example Entry

```json
{
  "image": "vol_01348_slice_57.png",
  "question": "Is there evidence of a lesion in the right kidney?",
  "answer": "No"
}

 
