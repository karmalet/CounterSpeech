# -*- coding: utf-8 -*-
"""
Prompts for GENERATING strategy-conditioned counterspeech from PANDA hate speech.

For each hate-speech sentence we generate one counterspeech per strategy:

- Logical Refutation     (逻辑反驳)
- Positive Reinterpretation (积极重构)
- Fact-Based Response    (事实回应)

The strategy definitions are kept consistent with prompts.py (the labeling step),
so that the generated text can later be verified by 02_label_panda_strategy.py.
"""

# Each strategy: a stable key (used for ids / filenames), the canonical English
# label (must match VALID_STRATEGIES in 02_label_panda_strategy.py), and a
# Chinese generation instruction describing how to write that kind of response.
STRATEGIES = [
    {
        "key": "logical_refutation",
        "name": "Logical Refutation",
        "name_zh": "逻辑反驳",
        "instruction": (
            "请主要通过【正面指出仇恨言论在推理上的矛盾与谬误】来反驳原文，"
            "揭示其逻辑漏洞。最常见也最有力的两个角度是："
            "（1）以偏概全：用个别人或个别事例去推断整个群体；"
            "（2）双重标准：一边反对自己被歧视，一边又理所当然地歧视他人。"
            "其他可用角度：个体不能代表群体、因果错误、内部自相矛盾、刻板印象的不合理性。"
            "重点在于拆解对方的论证结构，而不是诉诸情感、罗列事实或重构框架。"
        ),
    },
    {
        "key": "positive_reinterpretation",
        "name": "Positive Reinterpretation",
        "name_zh": "积极重构",
        "instruction": (
            "请主要通过【把仇恨言论中的负面标签或框架重新解释为中性、善意或更具建设性的视角】"
            "来回应，并用温和的语气缓和情绪对立。可用的角度包括："
            "指出换一种表达方式就能减少偏见、把被污名化的特征重新理解为无害甚至正面的东西、"
            "把对立的情绪场景转化为相互理解。"
            "重点在于『重新解释含义、降低敌意』，而不是直接指出逻辑错误或罗列事实。"
        ),
    },
    {
        "key": "fact_based_response",
        "name": "Fact-Based Response",
        "name_zh": "事实回应",
        "instruction": (
            "请主要通过【客观事实与具体事例，尤其是亲身经历和现实观察】"
            "来纠正原文中没有根据的主张或偏见。可用的角度包括："
            "『我去过／我接触过／我认识的人』这类第一人称的真实经历、"
            "可观察到的现实情况、对某个被滥用词语含义的澄清。"
            "重点在于用真实、具体的事实和经历说话，而不是只讲逻辑或只谈情绪。"
            "请不要编造具体的统计数字、研究或新闻来源；"
            "如果没有可靠数据，就用真实可信的个人经历和常识性事实。"
        ),
    },
]

STRATEGY_BY_KEY = {s["key"]: s for s in STRATEGIES}


GEN_SYSTEM_PROMPT = """你是一名专业的中文反仇恨言论（counterspeech）写作者。
你的任务是针对给定的中文仇恨言论，按照指定的【回应策略】撰写一条高质量的反仇恨言论。

写作要求：
1. 必须主要采用指定的策略，使该策略在回应中处于核心、主导地位。
2. 直接回应原文的具体内容，做到有的放矢，而不是泛泛而谈。
3. 语气理性、克制、有说服力，目标是改变旁观者和发言者的看法。
4. 不要复制、转述或强化原文中的侮辱性、歧视性内容。
5. 不要编造具体的统计数字、研究或新闻来源。
6. 长度控制在 1~4 句话之间，使用自然流畅的简体中文。
7. 只能输出严格 JSON，不要输出任何额外文字或解释。

输出格式：
{"counterspeech": "在此填写中文反仇恨言论"}
"""

GEN_USER_PROMPT_TEMPLATE = """请针对下面的仇恨言论，按照指定策略撰写一条反仇恨言论。

[回应策略]
{strategy_name_zh}（{strategy_name_en}）
{strategy_instruction}

[仇恨言论]
{hate_text}

请严格按照指定策略撰写，并按指定 JSON 格式输出。"""


# --------------------------------------------------------------------------- #
# Condition A (zero_shot) prompt — READY but NOT yet wired into
# generate_counterspeech.py. Based on README_시작점.md Day-2. Uses the same
# {"counterspeech": "..."} JSON contract as condition B.
# --------------------------------------------------------------------------- #
ZEROSHOT_SYSTEM_PROMPT = """你是一名专业的中文反仇恨言论（counterspeech）写作者。
针对给定的中文仇恨言论，撰写一条不具攻击性、有说服力的中文反仇恨言论。
要求：
1. 不要复制或重复原文的仇恨内容。
2. 不要再次贬低任何特定群体。
3. 长度控制在 1~2 句话。
4. 只输出严格 JSON，不要输出任何额外文字。
输出格式：{"counterspeech": "在此填写中文反仇恨言论"}
"""

ZEROSHOT_USER_TEMPLATE = """[仇恨言论]
{hate_text}

请按指定 JSON 格式输出反仇恨言论。"""
