"""肿瘤相关关键词，用于从通用医学 benchmark 中筛选肿瘤题。"""

from __future__ import annotations

# 中文肿瘤关键词
ZH_ONCOLOGY_KEYWORDS = [
    "肿瘤", "癌", "恶性", "良性肿瘤", "转移", "复发", "化疗", "放疗",
    "靶向", "免疫治疗", "新辅助", "辅助治疗",
    "腺癌", "鳞癌", "小细胞癌", "肉瘤", "淋巴瘤", "白血病",
    "肺癌", "胃癌", "肝癌", "结肠癌", "直肠癌", "乳腺癌", "卵巢癌",
    "前列腺癌", "胰腺癌", "食管癌", "甲状腺癌", "膀胱癌",
    "TNM", "分期", "EGFR", "ALK", "PD-L1", "HER2", "KRAS",
]

# 英文肿瘤关键词
EN_ONCOLOGY_KEYWORDS = [
    "cancer", "tumor", "tumour", "carcinoma", "sarcoma", "lymphoma", "leukemia",
    "malignant", "metastasis", "metastatic", "chemotherapy", "radiotherapy",
    "neoadjuvant", "adjuvant", "targeted therapy", "immunotherapy",
    "adenocarcinoma", "squamous cell", "small cell",
    "lung cancer", "gastric cancer", "liver cancer", "colon cancer",
    "breast cancer", "prostate cancer", "pancreatic cancer",
    "TNM", "staging", "EGFR", "ALK", "PD-L1", "HER2", "KRAS",
    "oncology", "oncologic",
]


def is_oncology_zh(text: str) -> bool:
    """判断中文文本是否与肿瘤相关。"""
    return any(kw in text for kw in ZH_ONCOLOGY_KEYWORDS)


def is_oncology_en(text: str) -> bool:
    """判断英文文本是否与肿瘤相关。"""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in EN_ONCOLOGY_KEYWORDS)


def is_oncology(text: str) -> bool:
    """中英文都检查。"""
    return is_oncology_zh(text) or is_oncology_en(text)


# 角色相关关键词（用于训练数据筛选时区分外科 vs 内科）
SURGEON_KEYWORDS = [
    "手术", "切除", "切口", "术中", "术后", "围手术期", "可切除", "不可切",
    "淋巴清扫", "根治术", "姑息术", "腹腔镜", "胸腔镜", "微创",
    "surgery", "surgical", "resection", "operation", "operative",
    "lymphadenectomy", "laparoscopic", "thoracoscopic",
]

MEDICAL_ONCOLOGIST_KEYWORDS = [
    "化疗", "靶向", "免疫", "新辅助", "辅助", "维持治疗", "姑息治疗",
    "顺铂", "紫杉醇", "卡铂", "培美曲塞", "吉非替尼", "厄洛替尼",
    "克唑替尼", "帕博利珠单抗", "PD-1", "PD-L1",
    "chemotherapy", "targeted", "immunotherapy", "cisplatin", "paclitaxel",
    "gefitinib", "erlotinib", "pembrolizumab",
]


def is_surgeon_topic(text: str) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in SURGEON_KEYWORDS)


def is_oncologist_topic(text: str) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in MEDICAL_ONCOLOGIST_KEYWORDS)
