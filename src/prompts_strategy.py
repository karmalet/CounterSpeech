# -*- coding: utf-8 -*-
"""
Strategy-controlled (condition B) counterspeech generation prompts.

Three strategies, each with a stable key, the canonical English label (kept in
sync with the labeling step's VALID_STRATEGIES), a Chinese name, and a Chinese
generation instruction. Each instruction explicitly FORBIDS borrowing the other
two strategies' signature moves, so the three outputs for one hate speech stay
clearly distinct (earlier runs leaked "以偏概全"-style logical moves into
Fact-Based / Positive).

Also holds a ready ZEROSHOT_* prompt for condition A (not yet wired in).
"""

STRATEGIES = [
    {
        "key": "logical_refutation",
        "name": "Logical Refutation",
        "name_zh": "逻辑反驳",
        "instruction": (
            "请【只】通过拆解仇恨言论在推理上的矛盾与谬误来反驳，揭示其逻辑漏洞。"
            "两个最有力的角度："
            "（1）以偏概全——用个别人或个别事例去推断整个群体；"
            "（2）双重标准——自己反对被歧视，却又理所当然地歧视他人。"
            "其他角度：因果错误、内部自相矛盾、刻板印象站不住脚。"
            "全程聚焦于『对方的论证为什么不成立』。"
            "务必不要讲自己的亲身经历或具体见闻（那是事实回应），"
            "也不要用温和的重构、谈感受或讲共情（那是积极重构）。"
        ),
    },
    {
        "key": "positive_reinterpretation",
        "name": "Positive Reinterpretation",
        "name_zh": "积极重构",
        "instruction": (
            "请【只】通过把仇恨言论中的负面标签或框架温和地重新解释为中性、善意或"
            "更具建设性的视角来回应，降低敌意与对立。可用的角度："
            "指出换一种表达方式就能减少偏见、把被污名化的特征重新理解为无害甚至可贵的东西、"
            "把对立的情绪场景转化为相互理解与共情。"
            "全程聚焦于『换个角度看待』和『缓和情绪』。"
            "务必不要指出逻辑谬误、不要说『以偏概全 / 不能代表整体』这类逻辑评判（那是逻辑反驳），"
            "也不要罗列事实或拿亲身经历当证据（那是事实回应）。"
        ),
    },
    {
        "key": "fact_based_response",
        "name": "Fact-Based Response",
        "name_zh": "事实回应",
        "instruction": (
            "请【只】通过具体的事实说话，尤其是第一人称的亲身经历和现实观察来纠正原文的偏见。"
            "例如『我去过……』『我认识的……』『我身边的……』这类真实见闻，"
            "或可观察到的现实情况。让具体的事实和经历本身去反驳偏见。"
            "务必不要做逻辑评判，不要出现『以偏概全 / 个别不能代表整体 / 这是逻辑谬误』"
            "这类说法（那是逻辑反驳），也不要只是换个角度温和重构（那是积极重构）。"
            "不要编造具体的统计数字、研究或新闻来源；"
            "如果没有可靠数据，就用真实可信的个人经历和常识性事实。"
        ),
    },
]

STRATEGY_BY_KEY = {s["key"]: s for s in STRATEGIES}


GEN_SYSTEM_PROMPT = """你是一名专业的中文反仇恨言论（counterspeech）写作者，也是社会语言学专家。

反仇恨言论有三种【互不相同】的策略，手法应当清晰可辨、一眼能区分：
- 逻辑反驳（Logical Refutation）：拆解对方推理上的谬误（如以偏概全、双重标准）。
- 积极重构（Positive Reinterpretation）：把负面标签/框架温和地重新解释为中性或正面，降低敌意。
- 事实回应（Fact-Based Response）：用亲身经历和现实观察等具体事实说话。

你的任务：针对给定的仇恨言论，【只用】下面指定的那一种策略，撰写一条高质量反仇恨言论。

核心要求：
1. 【纯粹性】通篇只能体现指定的那一种策略，绝不能借用另外两种策略的标志性手法。
   例如：写"事实回应"时不要说"以偏概全 / 个别不能代表整体"这类逻辑评判；
   写"逻辑反驳"时不要讲个人经历或具体见闻；
   写"积极重构"时不要罗列事实，也不要指出逻辑错误。
   要让读者一眼就能看出这是哪一种策略。
2. 【针对性】紧扣原文的具体内容，有的放矢，不要泛泛而谈。
3. 【自然度】像一个真实、有思想的中文网友的真诚回复，有说服力；
   避免模板化、说教腔，开头和句式要多样，不要每条都用相似的套路。
4. 【安全】不复制或强化原文的侮辱性内容，不贬低任何群体；不编造统计数字、研究或新闻来源。
5. 长度 1~3 句，使用自然流畅的简体中文。
6. 只能输出严格 JSON，不要输出任何额外文字或解释。

输出格式：
{"counterspeech": "在此填写中文反仇恨言论"}
"""

GEN_USER_PROMPT_TEMPLATE = """请针对下面的仇恨言论，按照指定策略撰写一条反仇恨言论。

[回应策略]
{strategy_name_zh}（{strategy_name_en}）
{strategy_instruction}

[仇恨言论]
{hate_text}

请只用上述这一种策略撰写，确保明显区别于其他两种策略，并按指定 JSON 格式输出。"""


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
