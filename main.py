# -*- coding: utf-8 -*-
"""
GBT_parse 项目入口 —— 反例审查评测
从 GBT_test.json 读取已构造好的反例文档，交由大模型审查，
逐条输出模型预测结果（error_type / dimension / section_number），
并与 ground truth 对比后流式写入评测结果文件。

评测三层指标：
  检测层：Precision / Recall     —— 能否发现问题
  定位层：Location Accuracy      —— 能否指向正确位置（section_number）
  诊断层：Diagnosis Accuracy     —— 能否说对问题类型（error_type）

用法示例：
    python main.py --input data/data_test_C.2.1/GBT_test.json
    python main.py --input data/data_test_C.2.1/GBT_test.json --output output/eval_result.json --backend azure
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import LLM_CONFIG, AZURE_LLM_CONFIG, LOG_CONFIG, RAG_CONFIG
from core.llm_client import OpenAILLMClient, AzureLLMClient
from core.rag import ChromaRAGStore
from utils.helpers import setup_logging

# ── 系统提示词 ───────────────────────────────────────────────────────────────
_REVIEW_DIMENSIONS = ["C2.1", "C2.2", "C3.1", "C3.2", "C3.3"]
_AGENT_USE_LOCAL_PREFILTER = True#是否使用本地筛选
_AGENT_PREFILTER_CONTEXT_LIMIT = 18000
NP_MCS_LAMBDA = 1.0
NP_MCS_ALPHA = 0.15
NP_MCS_BETA = 100.0

_VALID_ERROR_TYPES_BY_DIM: dict[str, set[str]] = {
    "C2.1": {"E-S-01", "E-S-02", "E-S-03", "E-S-04", "E-S-05", "E-S-06"},
    "C2.2": {"E-SC-01", "E-SC-02", "E-SC-03"},
    "C3.1": {"E-T-01", "E-T-02", "E-T-03", "E-T-04", "E-T-05", "E-T-06", "E-T-07"},
    "C3.2": {"E-TM-01", "E-TM-02", "E-TM-03", "E-TM-04", "E-TM-05"},
    "C3.3": {"E-R-01", "E-R-02", "E-R-03", "E-R-04"},
}

_HIGH_RECALL_ERROR_TYPE_AGENTS: list[tuple[str, str]] = [
    ("C3.1", "E-T-02"),
    ("C3.1", "E-T-07"),
    ("C3.2", "E-TM-01"),
    ("C3.2", "E-TM-02"),
    ("C3.2", "E-TM-03"),
    ("C3.2", "E-TM-04"),
    ("C3.3", "E-R-01"),
]

_ERROR_TYPE_EXPANSIONS: dict[tuple[str, str], list[str]] = {
    ("C2.1", "E-S-02"): ["E-S-03"],
    ("C2.1", "E-S-03"): ["E-S-02"],
    ("C3.1", "E-T-01"): ["E-T-02", "E-T-03", "E-T-07"],
    ("C3.1", "E-T-02"): ["E-T-01", "E-T-07"],
    ("C3.1", "E-T-03"): ["E-T-01", "E-T-07"],
    ("C3.1", "E-T-07"): ["E-T-01", "E-T-02", "E-T-03"],
    ("C3.2", "E-TM-01"): ["E-TM-03", "E-TM-04"],
    ("C3.2", "E-TM-03"): ["E-TM-01", "E-TM-04"],
    ("C3.2", "E-TM-04"): ["E-TM-01", "E-TM-03"],
    ("C3.3", "E-R-03"): ["E-R-01", "E-R-02"],
}

_SECTION_EXPAND_DIMS = {"C2.2", "C3.1", "C3.2", "C3.3"}
_MODAL_WORDS = ("不应", "不宜", "必须", "须", "应", "宜", "可", "尽量", "最好", "一般", "原则上", "必要时")

_SYSTEM_PROMPTS: dict[str, str] = {

    "ALL": """\
你是一名 GB/T 国家推荐性标准文档审查专家，专门负责依据 GB/T 1.1 审查标准文档是否存在错误。

请仔细阅读以下 GB/T 标准文档正文，同时审查 C2.1、C2.2、C3.1、C3.2、C3.3 五个维度下的所有错误。

错误维度说明：
- C2.1：文档结构类错误（章节缺失、顺序混乱、名称非标准、前言/引言混淆、附录归类错误、层级超限等）
- C2.2：范围一致性错误（范围与正文覆盖对象、适用边界、文件标题或范围章节表达原则不一致）
- C3.1：语气审查错误（“应”“宜”“可”“不应”“不宜”等助动词误用、强弱程度错误或模糊语气词误用）
- C3.2：术语审查错误（术语未定义、术语未使用、术语与定义不一致、同一概念多术语等）
- C3.3：规范性引用审查错误（漏列引用、冗余引用、引用无效、引用顺序或格式错误等）

错误类型说明：

C2.1 文档结构类：
- E-S-01：必备要素缺失（某必备章节被删除）
- E-S-02：章节顺序错误（章节位置被交换）
- E-S-03：非标准章节名称（章节标题使用了非标准写法）
- E-S-04：前言/引言混淆（前言与引言内容被互换）
- E-S-05：附录归类错误（规范性/资料性附录属性被互换）
- E-S-06：层级深度超限（章节编号层级超过 5 级）

C2.2 范围一致性类：
- E-SC-01：正文超出范围界定（正文规定了范围章节未提及的对象、场景或技术内容）
- E-SC-02：正文未覆盖范围承诺（范围声明适用于某类对象或场景，但正文无对应技术要求）
- E-SC-03：范围包含要求性条款（范围章节混入“应”“宜”等具体要求或建议）

C3.1 语气类：
- E-T-01：宜替换应（强制要求误用推荐语气）
- E-T-02：应替换宜（推荐性建议误用强制语气）
- E-T-03：可替换应（强制要求误用许可语气）
- E-T-04：混用应与须（使用“须”代替 GB/T 1.1 推荐的“应”）
- E-T-05：使用模糊语气词（“尽量”“最好”“一般”等替代规范助动词）
- E-T-06：不宜替换不应（禁止性要求误用不推荐语气）
- E-T-07：助动词与条款内容矛盾（明显强制安全或技术要求却使用“宜”或“可”）

C3.2 术语类：
- E-TM-01：使用未定义术语（正文出现技术术语，但术语和定义中没有定义）
- E-TM-02：术语未在正文使用（术语和定义新增术语，但正文从未使用）
- E-TM-03：术语与定义不一致（正文用法偏离术语定义）
- E-TM-04：同一概念多术语（同一概念在不同章节使用多个近义术语）
- E-TM-05  术语编号不连续(在「术语和定义」章节中删除一个编号，制造编号跳号)

C3.3 规范性引用类：
- E-R-01：漏列引用文件（正文规范性引用了某文件，但规范性引用文件清单未列出）
- E-R-02：冗余引用文件（清单列出文件，但正文从未规范性引用）
- E-R-03：资料性引用混入清单 : 仅在"注"或"示例"中提及的标准被列入规范性清单
- E-R-04：引用文件排列顺序错误（规范性引用文件清单顺序被打乱）

审查要求：
- 必须覆盖上述五个维度，不要只审查某一个维度。
- 输出所有你能确认的错误；同一处错误只输出一次。
- 如果没有发现错误，输出空数组 []。
- 至少输出20个错误。
- section_number 必须填写错误所在章节编号；无编号章节填 null，附录填附录编号如 "附录A"。
- reason 用一句话概括错误原因。
- error_description 需要更完整，建议按“错误：...；修改建议：...；依据：...”描述，便于后续评测。

请以 JSON 数组格式返回所有发现的错误，每个错误包含：
- "error_type"：错误类型代码（如 "E-S-01"）
- "dimension"：错误维度（如 "C2.1"）
- "section_number"：错误所在章节编号（无编号章节填 null，附录填附录编号如 "附录A"）
- "reason"：一句话说明错误原因
- "error_description"：对错误、修改建议和依据的完整说明

只输出 JSON 数组，不要输出任何其他内容。\
""",

}

_USER_PROMPT_TEMPLATE = """\
以下是待审查的 GB/T 标准文档正文：

---
{source_text}
---

请审查上述文档并输出发现的所有错误（JSON 数组）：\
"""

_RAG_USER_PROMPT_TEMPLATE = """\
以下是从本地 GB/T 审查知识库检索到的参考规则。它们只作为审查依据，不代表待审查文档一定存在这些错误。
请结合参考规则和待审查正文进行判断；如果参考规则与正文事实不一致，以正文事实为准。

【参考规则】
{rag_context}

以下是待审查的 GB/T 标准文档正文：

---
{source_text}
---

请审查上述文档并输出发现的所有错误（JSON 数组）：\
"""

# _AGENT_DIMENSION_GUIDES: dict[str, str] = {
#     "C2.1": """\
# 只审查 C2.1 文档结构类错误：
# - E-S-01：必备要素缺失
# - E-S-02：章节顺序错误
# - E-S-03：非标准章节名称
# - E-S-04：前言/引言混淆
# - E-S-05：附录归类错误
# - E-S-06：层级深度超限
# 重点查看章节目录、章条编号、标题名称、前言/引言、附录属性和编号层级。""",
#     "C2.2": """\
# 只审查 C2.2 范围一致性错误：
# - E-SC-01：正文超出范围界定
# - E-SC-02：正文未覆盖范围承诺
# - E-SC-03：范围包含要求性条款——在范围中植入"应/宜"等规范性语气

# 重点比较文件标题、范围章节、正文技术内容摘要和适用对象边界。""",
#     "C3.1": """\
# 只审查 C3.1 语气类错误：
# - E-T-01：宜替换应
# - E-T-02：应替换宜
# - E-T-03：可替换应
# - E-T-04：混用应与须
# - E-T-05：使用模糊语气词
# - E-T-06：不宜替换不应
# - E-T-07：助动词与条款内容矛盾
# 重点查看“应、宜、可、不应、不宜、须、必须、尽量、最好、一般”等表达。""",
#     "C3.2": """\
# 只审查 C3.2 术语类错误：
# - E-TM-01：使用未定义术语
# - E-TM-02：术语未在正文使用
# - E-TM-03：术语与定义不一致
# - E-TM-04：同一概念多术语
# 重点比较“术语和定义”章节与正文术语用法。""",
#     "C3.3": """\
# 只审查 C3.3 规范性引用类错误：
# - E-R-01：漏列引用文件
# - E-R-02：冗余引用文件
# - E-R-03：引用失效或版本错误
# - E-R-04：引用条目格式错误
# - E-R-05：注日期引用格式错误
# - E-R-06：引用文件排列顺序错误
# 重点比较“规范性引用文件”清单和正文中的规范性引用。""",
# }


# ── 评测指标计算 ───────────────────────────────────────────────────────────────

_AGENT_DIMENSION_GUIDES: dict[str, str] = {
    "C2.1": """\
你只审查 C2.1：标准结构与层次编排问题。不要输出其他维度的问题。

审查目标：
- 判断标准文本的总体结构、前置部分、正文层级、附录、参考文献等是否符合 GB/T 1.1 的结构要求。
- 重点检查“有无缺失、顺序是否错误、层级编号是否异常、附录/参考文献位置是否合理”。
- 必须把问题定位到最小可用 section_number；如果只能定位到附录或前置部分，也要给出对应编号或名称。

错误类型：
- E-S-01：必备结构要素缺失、重复或放置错误。例如封面、目次、前言、范围、规范性引用文件等应有而无，或出现在明显错误的位置。
- E-S-02：章节标题、章条编号或层级设置不符合标准结构规则。例如跳号、重复编号、父子层级不成立、条款编号粒度异常。
- E-S-03：范围、规范性引用文件、术语和定义等基础章节设置错误。例如章节内容与标题不匹配，或应独立成章却混入其他章节。
- E-S-04：附录设置错误。例如资料性/规范性附录标识缺失或错误，附录编号、标题、顺序、正文引用关系异常。
- E-S-05：参考文献、索引、附录后置材料的位置或格式错误。例如参考文献放在附录前后关系不符合规范，或与规范性引用文件混淆。
- E-S-06：图、表、公式、示例等结构单元的编号层级或归属异常，影响标准结构识别。

判定要点：
- 先看目录和标题序列，再看相关章节正文，不要只凭局部词语判断。
- 结构问题必须能说明“当前写法是什么、GB/T 1.1 期望是什么、差异在哪里”。
- 若只是 OCR 乱码、页眉页脚、分页残留，不构成结构错误，除非它导致章条层级或标题真实不可辨认。

输出要求：
- dimension 固定为 "C2.1"。
- error_type 只能从 E-S-01 到 E-S-06 中选择。
- reason 写简短判断依据。
- error_description 写成可人工复核的描述，包含当前 section 的标题/编号和具体结构问题。
- evidence 摘录最关键的原文短句或标题序列，不要整段复制。
- 至少输出10个错误。
""",

#====================================================================================

    "C2.2": """\
你只审查 C2.2：标准范围、适用对象、规范性内容边界与标准属性问题。不要输出其他维度的问题。

审查目标：
- 判断标准的范围、适用对象、规范性引用、术语和定义等基础内容是否清楚、准确、边界一致。
- 重点检查“范围是否过宽/过窄、对象是否前后不一致、规范性要求是否混入非规范性内容、引用关系是否支撑正文要求”。

错误类型：
- E-SC-01：范围或适用对象表述不清、不完整或与正文内容不一致。例如范围说适用于 A，但正文大量规定 B。
- E-SC-02：规范性引用文件、术语和定义等支撑性章节与正文要求不匹配。例如正文强制使用某术语或引用文件，但前文未给出支撑。
- E-SC-03：规范性内容边界错误。例如把资料性说明写成强制要求，或把应作为要求的内容放进注、示例、资料性附录等非规范位置。

判定要点：
- 优先检查“范围”“规范性引用文件”“术语和定义”“要求/试验方法/检验规则”等章之间的对应关系。
- 不要因为文字风格不够顺就判错；必须指出边界、对象、引用或规范性属性上的具体冲突。
- 对引用文件问题，区分“规范性引用文件”与“参考文献”，不要混为一类。

输出要求：
- dimension 固定为 "C2.2"。
- error_type 只能从 E-SC-01、E-SC-02、E-SC-03 中选择。
- section_number 填最能体现问题的章节；跨章节问题填主要矛盾所在章节。
- evidence 给出能证明边界/对象/引用冲突的短证据。
- 至少输出10个错误。
""",

#====================================================================================

    "C3.1": """\
你只审查 C3.1：标准条款用语、规范性表述、要求表达问题。不要输出其他维度的问题。

审查目标：
- 判断条款是否使用了符合 GB/T 1.1 的规范性用语。
- 重点检查“要求、推荐、允许、可能/能力、陈述”是否混用，以及一句话是否表达了清晰、可验证的技术要求。

错误类型：
- E-T-01：要求型条款用语错误。应表达必须满足的要求，却未使用明确的要求型表达，或强制程度不清。
- E-T-02：推荐型条款用语错误。应表达建议或推荐，却写成强制要求，或推荐程度不清。
- E-T-03：允许型条款用语错误。应表达允许、可选择事项，却写成要求或陈述，导致权限边界不清。
- E-T-04：能力/可能性表述错误。将“能、可能、可以”等能力或可能性表达误写成规范性要求，或反向混用。
- E-T-05：条款表述不可验证、不具可操作性。例如使用“适当、充分、必要时、合理”等模糊词却没有判定条件。
- E-T-06：同一条款中多个要求并列关系不清，条件、对象、动作、结果不完整，导致执行或检验无法落地。
- E-T-07：注、示例、说明性文字中包含实际要求，或正文要求被放入非规范性表达中。

判定要点：
- 每个候选都要识别“主体、动作、条件、结果/指标”是否完整。
- 不要把普通说明文字误判为要求；只有承担规范性约束的语句才输出。
- 发现混用时，说明应改为“应/宜/可/能”等哪类表达，但不要直接大段改写全文。

输出要求：
- dimension 固定为 "C3.1"。
- error_type 只能从 E-T-01 到 E-T-07 中选择。
- reason 说明属于哪类用语错误。
- error_description 必须指出原句中的问题词或问题结构。
- evidence 摘录问题句。
- 至少输出10个错误。
""",

#====================================================================================

    "C3.2": """\
你只审查 C3.2：技术要素、方法、指标、条件与结果的完整性和一致性问题。不要输出其他维度的问题。

审查目标：
- 判断技术要求、试验方法、检验规则、标志包装运输贮存等技术内容是否完整、对应、可执行。
- 重点检查“要求有没有方法支撑、方法有没有条件和步骤、指标有没有单位/限值、结果判定有没有规则”。

错误类型：
- E-TM-01：技术要求缺少明确对象、指标、限值、单位或条件，导致无法判断是否合格。
- E-TM-02：试验/检测方法缺少关键条件、设备、样品、步骤、计算或结果表达，导致无法复现。
- E-TM-03：要求与试验方法、检验规则之间不对应。例如有要求无方法，有方法无要求，或编号/项目不一致。
- E-TM-04：条件、参数、单位、符号或数值前后不一致，影响技术实施或判定。
- E-TM-05：多个技术要素之间逻辑顺序或依赖关系错误，例如先判定后试验、检验项目缺少抽样规则等。

判定要点：
- 优先检查“要求/试验方法/检验规则”三者是否闭环。
- 只有影响实施、验证、判定的缺陷才输出；普通措辞问题归 C3.1，不归 C3.2。
- 如果问题跨章节，section_number 填最直接暴露问题的章节，并在 reason 中说明关联章节。

输出要求：
- dimension 固定为 "C3.2"。
- error_type 只能从 E-TM-01 到 E-TM-05 中选择。
- evidence 应包含指标、方法、条件或判定规则相关的关键原文。
- 至少输出10个错误。
""",

#====================================================================================

    "C3.3": """\
你只审查 C3.3：规范性引用文件、参考文献、外部文件引用和引用一致性问题。不要输出其他维度的问题。

审查目标：
- 判断正文引用、规范性引用文件清单、参考文献之间是否一致、完整、格式正确。
- 重点检查“正文引用了但清单没有、清单列了但正文未规范性引用、年代号/编号/名称不一致、引用性质错误”。

错误类型：
- E-R-01：正文中规范性引用了文件，但“规范性引用文件”章未列出，或列出信息不完整。
- E-R-02：“规范性引用文件”章列出的文件在正文中未被规范性引用，或只是资料性提及。
- E-R-03：引用文件编号、年代号、名称、版本前后不一致。
- E-R-04：把参考文献当作规范性引用，或把规范性引用错误放入参考文献。
- E-R-05：引用方式不符合标准写法，例如引用语境、引用位置、文件标识格式不规范，影响识别。
- E-R-06：引用文件与条款要求不匹配，例如引用了错误领域、错误部分或无法支撑该条款要求的文件。

判定要点：
- 同时查看正文引用位置、规范性引用文件章、参考文献章，不要只看单处。
- 区分“资料性提及”和“规范性引用”：只有正文要求必须按某文件执行时，才应进入规范性引用文件。
- 如果只是 OCR 导致的个别字符损坏，不要直接判错；必须能看出引用关系或文件信息真实不一致。

输出要求：
- dimension 固定为 "C3.3"。
- error_type 只能从 E-R-01 到 E-R-06 中选择。
- section_number 填正文引用处或引用文件清单中最能定位问题的位置。
- evidence 同时给出引用处和清单/参考文献处的关键短证据；如只能找到一处，说明缺失的是另一处。
- 至少输出10个错误。
""",
}


def _normalize_section(raw) -> str | None:
    """
    将 section_number 归一化，使不同写法能正确匹配。

    归一化规则：
      "[B]"  / "附录 B" / "附录B" / "B" → "B"
      "[5.1]" / "5.1"                   → "5.1"
      None                              → None
    """
    if raw is None:
        return None
    s = str(raw).strip()
    # 去方括号：[B] → B, [5.1] → 5.1
    s = re.sub(r"^\[(.+)]$", r"\1", s)
    # 去"附录"前缀和空格：附录 A → A, 附录A → A
    s = re.sub(r"^附\s*录\s*", "", s)
    # 去残留空格
    return s.strip()


_DESCRIPTION_EQUIVALENCE_SYSTEM_PROMPT = """\
你是一名 GB/T 标准审查评测裁判。请判断两段错误描述表达的审查含义是否等价。

判定为等价的条件：
- 指向的是同一个错误事实；
- 错误类型、错误位置、修改建议或依据的核心含义一致；
- 允许措辞、详略、语序不同。

判定为不等价的条件：
- 描述的是不同错误；
- 关键位置、错误对象、错误原因、修改方向或依据不同；
- 预测描述过于笼统，不能覆盖标准答案的核心含义。

只输出 JSON：{"same": true} 或 {"same": false}。\
"""


def _description_text(item: dict) -> str:
    """兼容预测中的 error_description / reason 字段。"""
    return str(item.get("error_description") or item.get("reason") or "").strip()


def _descriptions_semantically_equal(gt_desc: str, pred_desc: str, llm) -> bool:
    """用 LLM 判断两段 error_description 是否语义等价。"""
    gt_desc = str(gt_desc or "").strip()
    pred_desc = str(pred_desc or "").strip()
    if not gt_desc or not pred_desc:
        return False

    gt_norm = re.sub(r"\s+", "", gt_desc)
    pred_norm = re.sub(r"\s+", "", pred_desc)
    if gt_norm == pred_norm:
        return True

    if llm is None:
        return False

    messages = [
        {"role": "system", "content": _DESCRIPTION_EQUIVALENCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "标准答案 error_description：\n"
                f"{gt_desc}\n\n"
                "模型预测 error_description：\n"
                f"{pred_desc}\n\n"
                "请判断二者表达的错误含义是否等价。"
            ),
        },
    ]
    try:
        raw = llm.chat(messages, temperature=0.0, max_tokens=256)
        parsed = llm.parse_json_response(raw)
        if isinstance(parsed, dict):
            return bool(parsed.get("same"))
        clean = str(raw).strip().lower()
        return '"same": true' in clean or '"same":true' in clean
    except Exception as exc:
        logging.getLogger(__name__).warning("error_description 语义等价判断失败: %s", exc)
        return False


def _all_gt_matched(
    ground_truth: list[dict],
    predictions: list[dict],
    predicate,
) -> bool:
    """要求 ground_truth 中每条反例都一对一匹配到一条 prediction。"""
    if len(predictions) < len(ground_truth):
        return False

    return _count_one_to_one_matches(ground_truth, predictions, predicate) >= len(ground_truth)


def _diagnosis_equal(gt: dict, pred: dict) -> bool:
    return (
        _normalize_section(gt.get("section_number")) == _normalize_section(pred.get("section_number"))
        and pred.get("error_type") == gt.get("error_type")
        and pred.get("dimension") == gt.get("dimension")
    )


def _diagnosis_match_key(item: dict) -> tuple[str | None, str | None, str | None]:
    return (
        _normalize_section(item.get("section_number")),
        item.get("dimension"),
        item.get("error_type"),
    )


def _counter_covers(required: Counter, available: Counter) -> bool:
    return all(available.get(key, 0) >= count for key, count in required.items())


def _count_one_to_one_matches(
    ground_truth: list[dict],
    predictions: list[dict],
    predicate,
) -> int:
    matched_gt_by_pred = [-1] * len(predictions)
    edge_cache: dict[tuple[int, int], bool] = {}

    def has_edge(gt_index: int, pred_index: int) -> bool:
        key = (gt_index, pred_index)
        if key not in edge_cache:
            edge_cache[key] = bool(predicate(ground_truth[gt_index], predictions[pred_index]))
        return edge_cache[key]

    def augment(gt_index: int, seen_pred: set[int]) -> bool:
        for pred_index in range(len(predictions)):
            if pred_index in seen_pred or not has_edge(gt_index, pred_index):
                continue
            seen_pred.add(pred_index)
            previous_gt = matched_gt_by_pred[pred_index]
            if previous_gt == -1 or augment(previous_gt, seen_pred):
                matched_gt_by_pred[pred_index] = gt_index
                return True
        return False

    matched_count = 0
    for gt_index in range(len(ground_truth)):
        if augment(gt_index, set()):
            matched_count += 1
    return matched_count


def judge_diagnosis_hit_counts(
    ground_truth: list[dict],
    predictions: list[dict],
    max_k: int = 10,
) -> tuple[int, dict[str, bool]]:
    gt_counts = Counter(_diagnosis_match_key(item) for item in ground_truth)
    pred_counts = Counter(_diagnosis_match_key(item) for item in predictions)
    matched_count = sum(min(count, pred_counts.get(key, 0)) for key, count in gt_counts.items())
    return matched_count, {f"H{k}": matched_count >= k for k in range(1, max_k + 1)}


def _item_match_key(item: dict, fields: tuple[str, ...]) -> tuple:
    values = []
    for field in fields:
        if field == "section_number":
            values.append(_normalize_section(item.get("section_number")))
        else:
            values.append(item.get(field))
    return tuple(values)


def count_item_matches(
    ground_truth: list[dict],
    predictions: list[dict],
    fields: tuple[str, ...],
) -> int:
    gt_counts = Counter(_item_match_key(item, fields) for item in ground_truth)
    pred_counts = Counter(_item_match_key(item, fields) for item in predictions)
    return sum(min(count, pred_counts.get(key, 0)) for key, count in gt_counts.items())


def _round_metric(value: float | None, digits: int = 6) -> float:
    return round(float(value), digits) if value is not None else 0.0


def _miss_count(metrics: dict) -> int:
    return max(0, int(metrics.get("ground_truth_count") or 0) - int(metrics.get("diagnosis_hit") or 0))


def _false_positive_proxy(metrics: dict) -> int:
    return max(0, int(metrics.get("prediction_count") or 0) - int(metrics.get("diagnosis_hit") or 0))


def _mcs_score(diagnosis_hit: int, ground_truth_count: int) -> float:
    if ground_truth_count <= 0:
        return 0.0
    recall = diagnosis_hit / ground_truth_count
    miss_rate = max(0, ground_truth_count - diagnosis_hit) / ground_truth_count
    return recall * math.exp(-NP_MCS_LAMBDA * miss_rate)


def _soft_noise_penalty(false_positive_proxy: int) -> float:
    fp = max(0.0, float(false_positive_proxy))
    if fp == 0:
        return 1.0
    return 1 - NP_MCS_ALPHA * fp / (fp + NP_MCS_BETA)


def _add_item_recall_derived_metrics(metrics: dict) -> None:
    gt_count = int(metrics.get("ground_truth_count") or 0)
    diagnosis_hit = int(metrics.get("diagnosis_hit") or 0)
    mcs = _mcs_score(diagnosis_hit, gt_count)
    metrics["miss_count"] = _miss_count(metrics)
    metrics["mcs_lambda_1"] = _round_metric(mcs)

    if "prediction_count" not in metrics:
        return

    fp = _false_positive_proxy(metrics)
    penalty = _soft_noise_penalty(fp)
    metrics["false_positive_proxy"] = fp
    metrics["soft_noise_penalty_alpha015_beta100"] = _round_metric(penalty)
    metrics["np_mcs_soft_alpha015_beta100"] = _round_metric(mcs * penalty)


def _item_recall_rates(metrics: dict) -> dict:
    gt_count = int(metrics.get("ground_truth_count") or 0)
    return {
        "location_recall": round(metrics.get("location_hit", 0) / gt_count, 4) if gt_count else 0.0,
        "dimension_recall": round(metrics.get("dimension_hit", 0) / gt_count, 4) if gt_count else 0.0,
        "diagnosis_recall": round(metrics.get("diagnosis_hit", 0) / gt_count, 4) if gt_count else 0.0,
    }


def build_item_recall_metrics(
    ground_truth: list[dict],
    predictions: list[dict],
    include_prediction_count: bool = True,
) -> dict:
    metrics = {
        "ground_truth_count": len(ground_truth),
        "location_hit": count_item_matches(ground_truth, predictions, ("section_number",)),
        "dimension_hit": count_item_matches(ground_truth, predictions, ("section_number", "dimension")),
        "diagnosis_hit": count_item_matches(
            ground_truth,
            predictions,
            ("section_number", "dimension", "error_type"),
        ),
    }
    if include_prediction_count:
        metrics["prediction_count"] = len(predictions)
    metrics.update(_item_recall_rates(metrics))
    _add_item_recall_derived_metrics(metrics)
    return metrics


def _accumulate_item_recall(total: Counter, metrics: dict) -> None:
    for field in (
        "ground_truth_count",
        "prediction_count",
        "location_hit",
        "dimension_hit",
        "diagnosis_hit",
    ):
        if field in metrics:
            total[field] += int(metrics.get(field) or 0)

    gt_count = int(metrics.get("ground_truth_count") or 0)
    if gt_count <= 0:
        return

    diagnosis_hit = int(metrics.get("diagnosis_hit") or 0)
    mcs = _mcs_score(diagnosis_hit, gt_count)
    total["doc_weighted_mcs_sum"] += mcs * gt_count

    if "prediction_count" in metrics:
        fp = _false_positive_proxy(metrics)
        penalty = _soft_noise_penalty(fp)
        total["soft_noise_penalty_weighted_sum"] += penalty * gt_count
        total["np_mcs_weighted_sum"] += mcs * penalty * gt_count


def _finalize_item_recall(total: Counter, include_prediction_count: bool = True) -> dict:
    metrics = {
        "ground_truth_count": int(total.get("ground_truth_count", 0)),
        "location_hit": int(total.get("location_hit", 0)),
        "dimension_hit": int(total.get("dimension_hit", 0)),
        "diagnosis_hit": int(total.get("diagnosis_hit", 0)),
    }
    if include_prediction_count:
        metrics["prediction_count"] = int(total.get("prediction_count", 0))
    metrics.update(_item_recall_rates(metrics))
    _add_item_recall_derived_metrics(metrics)

    gt_count = metrics["ground_truth_count"]
    if gt_count:
        metrics["doc_weighted_mcs_lambda_1"] = _round_metric(
            total.get("doc_weighted_mcs_sum", 0.0) / gt_count,
        )
        if include_prediction_count:
            metrics["avg_soft_noise_penalty_alpha015_beta100"] = _round_metric(
                total.get("soft_noise_penalty_weighted_sum", 0.0) / gt_count,
            )
            metrics["doc_weighted_np_mcs_soft_alpha015_beta100"] = _round_metric(
                total.get("np_mcs_weighted_sum", 0.0) / gt_count,
            )
    return metrics


def _group_key(value) -> str:
    return "<NULL>" if value is None else str(value)


def update_item_recall_group_totals(
    totals: dict[str, Counter],
    ground_truth: list[dict],
    predictions: list[dict],
    group_field: str,
    metadata: dict[str, dict] | None = None,
) -> None:
    group_values = sorted({_group_key(item.get(group_field)) for item in ground_truth})
    for group_value in group_values:
        group_truth = [
            item
            for item in ground_truth
            if _group_key(item.get(group_field)) == group_value
        ]
        metrics = build_item_recall_metrics(
            group_truth,
            predictions,
            include_prediction_count=False,
        )
        if group_value not in totals:
            totals[group_value] = Counter()
        _accumulate_item_recall(totals[group_value], metrics)

        if metadata is not None and group_field == "error_type" and group_truth:
            dims = sorted({
                str(item.get("dimension"))
                for item in group_truth
                if item.get("dimension") is not None
            })
            if dims:
                metadata.setdefault(group_value, {})["dimension"] = dims[0] if len(dims) == 1 else ",".join(dims)


def finalize_item_recall_groups(
    totals: dict[str, Counter],
    metadata: dict[str, dict] | None = None,
) -> dict:
    result = {}
    for group_value in sorted(totals):
        group_metrics = _finalize_item_recall(totals[group_value], include_prediction_count=False)
        if metadata and group_value in metadata:
            group_metrics = {**metadata[group_value], **group_metrics}
        result[group_value] = group_metrics
    return result


def judge_doc(ground_truth: list[dict], predictions: list[dict], llm=None) -> tuple[bool, bool, bool]:
    """
    对单篇文档做文档级二值判定，返回 (precision_hit, location_hit, diagnosis_hit)。

    ground_truth 中每一条反例都必须在 predictions 中找到一对一匹配，才算命中：
      - Location（定位）:   section_number 相等
      - Diagnosis（检测）:  error_type + dimension + section_number 均相等
      - Precision（诊断）:  Diagnosis 命中，且 error_description 语义等价

    section_number 比较前会做归一化（去方括号、去"附录"前缀等）。
    """
    if not ground_truth:
        return True, True, True

    logger = logging.getLogger(__name__)
    logger.info(
        "[eval-match] start: ground_truth=%d predictions=%d",
        len(ground_truth),
        len(predictions),
    )

    gt_section_counts = Counter(_normalize_section(item.get("section_number")) for item in ground_truth)
    pred_section_counts = Counter(_normalize_section(item.get("section_number")) for item in predictions)
    location_hit = _counter_covers(gt_section_counts, pred_section_counts)
    logger.info("[eval-match] location done: hit=%s", location_hit)

    gt_diagnosis_counts = Counter(_diagnosis_match_key(item) for item in ground_truth)
    pred_diagnosis_counts = Counter(_diagnosis_match_key(item) for item in predictions)
    diagnosis_hit = _counter_covers(gt_diagnosis_counts, pred_diagnosis_counts)
    logger.info("[eval-match] diagnosis done: hit=%s", diagnosis_hit)

    description_cache: dict[tuple[str, str], bool] = {}

    def precision_eq(gt: dict, pred: dict) -> bool:
        gt_desc = _description_text(gt)
        pred_desc = _description_text(pred)
        cache_key = (gt_desc, pred_desc)
        if cache_key not in description_cache:
            description_cache[cache_key] = _descriptions_semantically_equal(gt_desc, pred_desc, llm)
        return description_cache[cache_key]

    precision_hit = False
    if not diagnosis_hit:
        logger.info("[eval-match] precision skipped: diagnosis_hit=False")
    else:
        gt_groups: dict[tuple[str | None, str | None, str | None], list[dict]] = {}
        pred_groups: dict[tuple[str | None, str | None, str | None], list[dict]] = {}
        for item in ground_truth:
            gt_groups.setdefault(_diagnosis_match_key(item), []).append(item)
        for item in predictions:
            pred_groups.setdefault(_diagnosis_match_key(item), []).append(item)

        precision_hit = True
        for key, gt_group in gt_groups.items():
            pred_group = pred_groups.get(key, [])
            logger.info(
                "[eval-match] precision group: key=%s ground_truth=%d predictions=%d",
                key,
                len(gt_group),
                len(pred_group),
            )
            if len(pred_group) < len(gt_group):
                precision_hit = False
                break

            matched_count = _count_one_to_one_matches(gt_group, pred_group, precision_eq)
            if matched_count < len(gt_group):
                precision_hit = False
                break
    logger.info("[eval-match] precision done: hit=%s", precision_hit)

    return precision_hit, location_hit, diagnosis_hit


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_test_json(path: str) -> dict:
    """读取 GBT_test.json，返回原始字典。"""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if "Invalid control character" not in str(exc):
            raise
        logging.getLogger(__name__).warning(
            "JSON 中包含未转义控制字符，使用 strict=False 兼容读取: %s", exc
        )
        return json.loads(text, strict=False)


def _build_rag_query(source_text: str, dim: str, limit: int = 5000) -> str:
    """构造用于检索审查规则的短查询，避免直接用超长全文做 embedding。"""
    compact = re.sub(r"\s+", " ", source_text).strip()
    dim_text = "、".join(_REVIEW_DIMENSIONS) if dim == "ALL" else dim
    return f"审查维度：{dim_text}\n待审查文档片段：{compact[:limit]}"


def _compact_text(text: str, limit: int = 5000) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact[:limit]


def _parse_review_sections(source_text: str) -> list[dict]:
    """从 source_text 中按 [section_number] title 形式切出章节，供 agentic 分步审查使用。"""
    sections: list[dict] = []
    current: dict | None = None

    def flush_current() -> None:
        nonlocal current
        if current is not None:
            current["content"] = "\n".join(current["content_lines"]).strip()
            current.pop("content_lines", None)
            sections.append(current)
            current = None

    for line in source_text.splitlines():
        stripped = line.strip()
        heading_match = re.match(r"^\[([^\]]+)]\s*(.*)$", stripped)
        if heading_match:
            flush_current()
            current = {
                "section_number": heading_match.group(1).strip(),
                "title": heading_match.group(2).strip(),
                "content_lines": [],
            }
            continue

        if stripped in {"前言", "前  言", "引言", "引  言", "参考文献", "目次", "目录"}:
            flush_current()
            current = {
                "section_number": None,
                "title": stripped,
                "content_lines": [],
            }
            continue

        if current is not None:
            current["content_lines"].append(line)

    flush_current()
    return sections


def _section_line(section: dict, content_limit: int = 260) -> str:
    number = section.get("section_number")
    title = section.get("title") or ""
    label = f"[{number}] {title}" if number else title
    return f"{label}\n{_compact_text(section.get('content', ''), content_limit)}"


def _build_agent_dimension_context(source_text: str, dim: str, limit: int = 12000) -> str:
    """按维度压缩正文，避免 agentic 每一步都喂完整长文。"""
    sections = _parse_review_sections(source_text)
    if not sections:
        return _compact_text(source_text, limit)

    heading_lines = [
        f"[{s.get('section_number')}] {s.get('title')}" if s.get("section_number") else str(s.get("title") or "")
        for s in sections
    ]

    if dim == "C2.1":
        focus = [
            _section_line(s, 500)
            for s in sections
            if any(key in str(s.get("title") or "") for key in ("前言", "引言", "附录", "范围", "规范性引用文件", "术语"))
            or (s.get("section_number") and str(s.get("section_number")).count(".") >= 4)
        ]
        context = "【章节标题列表】\n" + "\n".join(heading_lines) + "\n\n【重点章节】\n" + "\n\n".join(focus)
    elif dim == "C2.2":
        focus = [
            _section_line(s, 420)
            for s in sections
            if any(key in str(s.get("title") or "") for key in ("范围", "试验", "要求", "判定", "方法", "报告"))
            or (s.get("section_number") and str(s.get("section_number")).split(".")[0] in {"1", "4", "5", "6", "7", "8", "9"})
        ]
        context = "\n\n".join(focus)
    elif dim == "C3.1":
        modal_pat = re.compile(r"应|宜|可|不应|不宜|须|必须|尽量|最好|一般")
        focus = [_section_line(s, 520) for s in sections if modal_pat.search(str(s.get("content") or ""))]
        context = "\n\n".join(focus)
    elif dim == "C3.2":
        focus = [
            _section_line(s, 600)
            for s in sections
            if any(key in str(s.get("title") or "") for key in ("术语", "定义"))
            or str(s.get("section_number") or "").startswith("3")
        ]
        if len(focus) < 8:
            focus.extend(_section_line(s, 220) for s in sections[:20])
        context = "\n\n".join(focus)
    elif dim == "C3.3":
        ref_pat = re.compile(r"\b(?:GB/T|GB|ISO/IEC|ISO|IEC|DL/T|HG/T|SN/T|GA/T|JJF|JJG|YD/T|JB/T|IEEE|ASTM)\s*[\dA-Z]")
        focus = [
            _section_line(s, 650)
            for s in sections
            if "规范性引用文件" in str(s.get("title") or "") or ref_pat.search(str(s.get("content") or ""))
        ]
        context = "\n\n".join(focus)
    else:
        context = _compact_text(source_text, limit)

    return context[:limit]


_C31_HIGH_RISK_MODAL_KEYWORDS = (
    "不应", "不宜", "必须", "须", "尽量", "最好", "一般", "原则上", "必要时", "适当", "可", "宜",
)
_C31_ORDINARY_MODAL_KEYWORDS = ("应",)
_REF_PATTERN = re.compile(
    r"\b(?:GB/T|GB|ISO/IEC|ISO|IEC|DL/T|HG/T|SN/T|GA/T|JJF|JJG|YD/T|JB/T|IEEE|ASTM)\s*[/A-Z0-9.]*\s*\d+(?:\.\d+)*(?:[-—–]\d{4})?",
    re.I,
)


def _section_heading_lines(sections: list[dict], limit: int = 240) -> list[str]:
    return [
        f"[{s.get('section_number')}] {s.get('title')}" if s.get("section_number") else str(s.get("title") or "")
        for s in sections[:limit]
    ]


def _normalize_reference_id(ref: str) -> str:
    normalized = str(ref or "").upper()
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.strip("，。；;,.、()（）")


def _extract_reference_ids(text: str) -> set[str]:
    return {
        _normalize_reference_id(match.group(0))
        for match in _REF_PATTERN.finditer(str(text or ""))
        if match.group(0)
    }


def _build_c31_modal_context(source_text: str, limit: int = _AGENT_PREFILTER_CONTEXT_LIMIT) -> str:
    sections = _parse_review_sections(source_text)
    if not sections:
        return _compact_text(source_text, limit)

    high_risk_focus: list[str] = []
    ordinary_ying_focus: list[str] = []
    for section in sections:
        content = str(section.get("content") or "")
        section_text = _section_line(section, 900)
        if any(keyword in content for keyword in _C31_HIGH_RISK_MODAL_KEYWORDS):
            high_risk_focus.append(section_text)
        elif any(keyword in content for keyword in _C31_ORDINARY_MODAL_KEYWORDS):
            ordinary_ying_focus.append(section_text)

    context = (
        "【全文章节目录】\n"
        + "\n".join(_section_heading_lines(sections))
        + "\n\n【C3.1 高风险语气候选章节：含必须/须/宜/可/不应/不宜/模糊语气词等】\n"
        + ("\n\n".join(high_risk_focus) if high_risk_focus else "（未检出高风险语气候选章节）")
        + "\n\n【C3.1 普通“应”候选章节：优先级较低，仅用于补充核对】\n"
        + ("\n\n".join(ordinary_ying_focus) if ordinary_ying_focus else "（未检出仅含普通“应”的候选章节）")
    )
    return context[:limit]


def _build_c33_reference_context(source_text: str, limit: int = _AGENT_PREFILTER_CONTEXT_LIMIT) -> str:
    sections = _parse_review_sections(source_text)
    if not sections:
        return _compact_text(source_text, limit)

    ref_sections: list[str] = []
    ref_section_texts: list[str] = []
    body_ref_sections: list[str] = []
    body_ref_texts: list[str] = []

    for section in sections:
        title = str(section.get("title") or "")
        content = str(section.get("content") or "")
        text = f"{title}\n{content}"
        if "规范性引用文件" in title:
            ref_sections.append(_section_line(section, 2500))
            ref_section_texts.append(text)
        elif _REF_PATTERN.search(text):
            body_ref_sections.append(_section_line(section, 900))
            body_ref_texts.append(text)

    list_refs = set().union(*(_extract_reference_ids(text) for text in ref_section_texts)) if ref_section_texts else set()
    body_refs = set().union(*(_extract_reference_ids(text) for text in body_ref_texts)) if body_ref_texts else set()
    missing_refs = sorted(body_refs - list_refs)
    redundant_refs = sorted(list_refs - body_refs)

    context = (
        "【本地引用抽取结果】\n"
        f"清单引用：{', '.join(sorted(list_refs)) if list_refs else '（未检出）'}\n"
        f"正文引用：{', '.join(sorted(body_refs)) if body_refs else '（未检出）'}\n"
        f"疑似漏列（正文有、清单无）：{', '.join(missing_refs) if missing_refs else '（未检出）'}\n"
        f"疑似冗余（清单有、正文无）：{', '.join(redundant_refs) if redundant_refs else '（未检出）'}\n"
        "\n【规范性引用文件章节】\n"
        + ("\n\n".join(ref_sections) if ref_sections else "（未找到规范性引用文件章节）")
        + "\n\n【正文中含标准引用的章节】\n"
        + ("\n\n".join(body_ref_sections) if body_ref_sections else "（未检出正文引用章节）")
    )
    return context[:limit]


_C32_TECHNICAL_SECTION_TITLE_KEYWORDS = (
    "要求", "技术要求", "试验方法", "检验规则", "检测方法", "判定规则", "标志", "包装", "运输", "贮存",
)
_C32_TECHNICAL_CONTENT_KEYWORDS = (
    "限值", "指标", "单位", "试样", "样品", "设备", "步骤", "计算", "结果", "判定", "应符合", "按", "进行",
)
_C32_TERM_CONTENT_KEYWORDS = ("定义", "称为", "是指", "表示", "用于", "由", "组成")


def _looks_like_c32_technical_element(text: str) -> bool:
    text = str(text or "")
    return any(keyword in text for keyword in _C32_TECHNICAL_CONTENT_KEYWORDS + _C32_TERM_CONTENT_KEYWORDS)


def _build_c32_technical_element_context(source_text: str, limit: int = _AGENT_PREFILTER_CONTEXT_LIMIT) -> str:
    sections = _parse_review_sections(source_text)
    if not sections:
        return _compact_text(source_text, limit)

    term_sections: list[str] = []
    technical_title_sections: list[str] = []
    technical_keyword_sections: list[str] = []
    term_numbers: list[str] = []

    for section in sections:
        section_number = str(section.get("section_number") or "")
        title = str(section.get("title") or "")
        content = str(section.get("content") or "")
        text = f"{title}\n{content}"
        is_term_section = ("术语" in title and "定义" in title) or section_number == "3" or section_number.startswith("3.")
        is_technical_title_section = any(keyword in title for keyword in _C32_TECHNICAL_SECTION_TITLE_KEYWORDS)

        if is_term_section:
            term_sections.append(_section_line(section, 2500 if section_number in {"3", ""} else 800))
            if section_number and section_number != "3":
                term_numbers.append(section_number)
        elif is_technical_title_section:
            technical_title_sections.append(_section_line(section, 1200))
        elif _looks_like_c32_technical_element(text):
            technical_keyword_sections.append(_section_line(section, 700))

    context = (
        "【术语编号序列】\n"
        + (", ".join(term_numbers) if term_numbers else "（未检出术语编号）")
        + "\n\n【术语和定义相关章节】\n"
        + ("\n\n".join(term_sections) if term_sections else "（未找到术语和定义章节）")
        + "\n\n【C3.2 技术要素重点章节：标题含要求/试验方法/检验规则/标志包装运输贮存等】\n"
        + ("\n\n".join(technical_title_sections) if technical_title_sections else "（未检出技术要素标题章节）")
        + "\n\n【C3.2 技术要素关键词章节：含限值/指标/单位/试样/设备/步骤/计算/判定等】\n"
        + ("\n\n".join(technical_keyword_sections[:30]) if technical_keyword_sections else "（未检出技术要素关键词章节）")
    )
    return context[:limit]


def _build_agent_prefilter_context(source_text: str, dim: str, limit: int = _AGENT_PREFILTER_CONTEXT_LIMIT) -> str:
    if dim == "C3.1":
        return _build_c31_modal_context(source_text, limit)
    if dim == "C3.3":
        return _build_c33_reference_context(source_text, limit)
    if dim == "C3.2":
        return _build_c32_technical_element_context(source_text, limit)
    return _build_agent_dimension_context(source_text, dim, limit)


def _retrieve_rag_context(
    source_text: str,
    dim: str,
    rag_store: ChromaRAGStore | None,
    rag_top_k: int,
) -> str:
    if rag_store is None:
        return ""

    rag_query = _build_rag_query(source_text, dim)
    if dim == "ALL":
        rag_items = []
        for review_dim in _REVIEW_DIMENSIONS:
            rag_items.extend(rag_store.retrieve(rag_query, dim=review_dim, top_k=rag_top_k))
    else:
        rag_items = rag_store.retrieve(rag_query, dim=dim, top_k=rag_top_k)

    logging.getLogger(__name__).info("RAG 检索到 %d 条 %s 规则", len(rag_items), dim)
    return rag_store.format_context(rag_items)


def _parse_llm_json_list(raw: str, llm) -> list[dict] | None:
    result = llm.parse_json_response(raw)
    if result is None:
        return None
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def _normalize_prediction_item(item: dict) -> dict | None:
    error_type = str(item.get("error_type") or "").strip()
    dimension = str(item.get("dimension") or "").strip()
    if not error_type or not dimension:
        return None

    section_number = item.get("section_number")
    reason = str(item.get("reason") or item.get("error_description") or "").strip()
    error_description = str(item.get("error_description") or reason).strip()
    return {
        "error_type": error_type,
        "dimension": dimension,
        "section_number": section_number,
        "reason": reason,
        "error_description": error_description,
        **({"evidence": str(item.get("evidence")).strip()} if item.get("evidence") else {}),
        **({"confidence": item.get("confidence")} if item.get("confidence") is not None else {}),
        **({"critic_note": str(item.get("critic_note")).strip()} if item.get("critic_note") else {}),
    }


def _to_prediction_output(item: dict) -> dict | None:
    normalized = _normalize_prediction_item(item)
    if normalized is None:
        return None
    return {
        "error_type": normalized["error_type"],
        "dimension": normalized["dimension"],
        "section_number": normalized.get("section_number"),
        "reason": normalized["reason"],
        "error_description": normalized["error_description"],
    }


def _dedupe_predictions(predictions: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in predictions:
        normalized = _normalize_prediction_item(item)
        if normalized is None:
            continue
        key = (
            normalized["dimension"],
            normalized["error_type"],
            _normalize_section(normalized.get("section_number")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _prediction_match_key(item: dict) -> tuple[str, str, str | None] | None:
    normalized = _normalize_prediction_item(item)
    if normalized is None:
        return None
    return (
        normalized["dimension"],
        normalized["error_type"],
        _normalize_section(normalized.get("section_number")),
    )


def _confidence_value(item: dict) -> float:
    try:
        value = float(item.get("confidence"))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _filter_agent_supplements(
    candidates: list[dict],
    direct_predictions: list[dict],
    confidence_threshold: float = 0.75,
    max_per_dim: int = 5,
) -> list[dict]:
    direct_keys = {
        key
        for key in (_prediction_match_key(item) for item in direct_predictions)
        if key is not None
    }
    grouped: dict[str, list[dict]] = {dim: [] for dim in _REVIEW_DIMENSIONS}
    seen_candidate_keys: set[tuple[str, str, str | None]] = set()

    for candidate in _dedupe_predictions(candidates):
        dim = candidate["dimension"]
        error_type = candidate["error_type"]
        if error_type not in _VALID_ERROR_TYPES_BY_DIM.get(dim, set()):
            continue
        if _confidence_value(candidate) < confidence_threshold:
            continue

        key = _prediction_match_key(candidate)
        if key is None or key in direct_keys or key in seen_candidate_keys:
            continue

        seen_candidate_keys.add(key)
        grouped.setdefault(dim, []).append(candidate)

    supplements: list[dict] = []
    for dim in _REVIEW_DIMENSIONS:
        ranked = sorted(
            grouped.get(dim, []),
            key=lambda item: _confidence_value(item),
            reverse=True,
        )
        supplements.extend(ranked[:max_per_dim])
    return supplements


def call_llm_review(
    source_text: str,
    llm,
    dim: str,
    rag_store: ChromaRAGStore | None = None,
    rag_top_k: int = 5,
) -> list[dict] | None:
    """
    将 source_text 送入 LLM，返回模型预测的错误列表。
    每条格式：{"error_type", "dimension", "section_number", "reason"}
    dim 用于从 _SYSTEM_PROMPTS 中选取对应维度的 system prompt。
    """
    system_prompt = _SYSTEM_PROMPTS.get(dim)
    if system_prompt is None:
        raise ValueError(f"未找到维度 '{dim}' 对应的 system prompt，请先在 _SYSTEM_PROMPTS 中添加。")

    rag_context = _retrieve_rag_context(source_text, dim, rag_store, rag_top_k)

    user_prompt = (
        _RAG_USER_PROMPT_TEMPLATE.format(rag_context=rag_context, source_text=source_text)
        if rag_context
        else _USER_PROMPT_TEMPLATE.format(source_text=source_text)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    raw = llm.chat(messages, temperature=0.0, max_tokens=32000)
    result = llm.parse_json_response(raw)
    if result is None:
        return None
    if isinstance(result, list):
        return result
    return []


def _agent_scan_dimension(
    source_text: str,
    llm,
    dim: str,
    rag_store: ChromaRAGStore | None = None,
    rag_top_k: int = 5,
    judge_feedback: str = "",
    known_predictions: list[dict] | None = None,
    scan_mode: str = "supplement",
) -> list[dict] | None:
    """Specialist Agent：单维度扫描候选错误。"""
    use_local_prefilter = _AGENT_USE_LOCAL_PREFILTER
    focused_context = _build_agent_prefilter_context(source_text, dim) if use_local_prefilter else ""
    focused_context_block = (
        f"""\
【本地预筛重点上下文】
以下内容由本地规则预筛得到，用于提示最可能相关的章节、引用集合或术语线索。
请先重点审查这一部分；发现疑似错误后，再结合完整正文确认上下文、章节号和错误类型。
{focused_context}
"""
        if use_local_prefilter
        else "【本地预筛重点上下文】\n（未启用本地预筛，请直接结合完整正文审查。）"
    )
    rag_context = _retrieve_rag_context(source_text, dim, rag_store, rag_top_k)
    global_guide = _SYSTEM_PROMPTS["ALL"]
    guide = _AGENT_DIMENSION_GUIDES.get(dim, "")
    valid_error_types = _VALID_ERROR_TYPES_BY_DIM.get(dim, set())
    valid_error_types_text = "、".join(sorted(valid_error_types))
    high_recall_mode = scan_mode == "full_scan"
    known_predictions_json = json.dumps(
        [
            item
            for item in (_to_prediction_output(prediction) for prediction in _dedupe_predictions(known_predictions or []))
            if item is not None
        ],
        ensure_ascii=False,
        indent=2,
    )
    recall_rule_text = (
        "5. 当前任务以最大化召回率为目标：只要某章节存在可复核证据、可能构成当前维度错误，就输出候选；允许中等置信度候选，不要因担心误报而漏掉可疑项。"
        if high_recall_mode
        else "5. 证据不足时不要输出，不要为了凑数量输出低置信度结果。"
    )

    system_prompt = f"""\
你是一名 GB/T 标准审查 Specialist Agent。
你会看到完整错误维度体系，这是为了帮助你区分相邻维度、避免把其他维度的问题误归类到当前维度。

【完整错误维度体系】
下面的 ALL 规则只作为错误类型、维度边界和判定标准参考。
其中“必须覆盖五个维度”“至少输出20个错误”等面向 direct 审查的要求，不适用于当前 Specialist。
{global_guide}

【当前重点审查维度】
当前你只负责审查：{dim}

{guide}

【强制规则】
1. 只能输出 dimension == "{dim}" 的错误。
2. error_type 只能从以下集合中选择：{valid_error_types_text}。
3. 如果某个问题更适合归入其他维度，即使它看起来存在，也不要输出。
4. section_number 必须使用文档中的实际章节号；无法确定章节号时填 null。
{recall_rule_text}

输出 JSON 数组。每个元素必须包含：
- "error_type"
- "dimension"
- "section_number"
- "reason"
- "error_description"
- "evidence"：支持该判断的原文证据片段或章节摘要
- "confidence"：0 到 1 之间的小数，表示你对该候选错误成立的信心

只输出 JSON 数组，不要输出解释。"""

    if high_recall_mode:
        user_prompt = f"""\
【参考规则】
{rag_context or "（无）"}

{focused_context_block}

【待审查完整正文】
以下是完整文档正文，用于核对本地预筛上下文是否遗漏相关证据。
{source_text}

【任务】
请对 {dim} 进行独立全量审查。你的目标是提高召回率：
- 不要只做补漏，不要依赖已有主审结果；
- 对当前维度下所有可能成立的错误都输出候选；
- 同一章节如果可能对应多个 error_type，可以分别输出；
- 对跨章节问题，优先输出最可能被人工标注为错误位置的 section_number，同时也可以输出关键对照章节。

若没有候选错误，输出 []。"""
    else:
        user_prompt = f"""\
【参考规则】
{rag_context or "（无）"}

{focused_context_block}

【待审查完整正文】
以下是完整文档正文，用于核对本地预筛上下文是否遗漏相关证据。
请不要在完整正文中泛化搜索低置信错误；只有在证据充分、且符合当前维度时才输出。
{source_text}

【已有主审结果】
{known_predictions_json or "[]"}

【上一轮 Judge 反馈】
{judge_feedback or "（无）"}

请只审查 {dim}，重点查找已有主审结果可能遗漏的错误。
不要重复输出已有主审结果中相同 section_number + dimension + error_type 的错误。
若没有新的高置信候选错误，输出 []。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=8000,
    )
    parsed = _parse_llm_json_list(raw, llm)
    if parsed is None:
        return None
    items = [item for item in (_normalize_prediction_item(x) for x in parsed) if item is not None]
    return [
        item
        for item in items
        if item["dimension"] == dim and item["error_type"] in valid_error_types
    ]


def _agent_verify_candidates(
    source_text: str,
    llm,
    dim: str,
    candidates: list[dict],
) -> list[dict] | None:
    """Agentic 第二步：对单维度候选错误进行复核，删除证据不足和误报。"""
    if not candidates:
        return []

    dim_context = _build_agent_dimension_context(source_text, dim, limit=15000)
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    system_prompt = """\
你是一名 GB/T 标准审查复核 agent。
请基于给定文档证据复核候选错误是否真实成立。
只保留证据充分、错误类型和章节位置都合理的候选项。
如果候选项成立，可以润色 reason 和 error_description，但不要改变其核心含义。
输出 JSON 数组；每个元素必须包含 error_type、dimension、section_number、reason、error_description。
只输出 JSON 数组，不要输出解释。"""

    user_prompt = f"""\
【审查维度】
{dim}

【文档证据】
{dim_context}

【候选错误】
{candidates_json}

请删除不成立、证据不足、维度不符或重复的候选错误，返回复核后的 JSON 数组。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=8000,
    )
    parsed = _parse_llm_json_list(raw, llm)
    if parsed is None:
        return None
    return [item for item in (_normalize_prediction_item(x) for x in parsed) if item is not None]


def _build_multi_agent_evidence_context(
    source_text: str,
    candidates: list[dict],
    limit: int = 22000,
) -> str:
    """给 Critic/Arbiter 构造紧凑证据上下文：标题树 + 候选所在章节 + 维度片段。"""
    sections = _parse_review_sections(source_text)
    if not sections:
        return _compact_text(source_text, limit)

    heading_lines = [
        f"[{s.get('section_number')}] {s.get('title')}" if s.get("section_number") else str(s.get("title") or "")
        for s in sections
    ]
    section_by_num = {
        _normalize_section(s.get("section_number")): s
        for s in sections
        if s.get("section_number") is not None
    }

    snippets: list[str] = []
    seen_sections: set[str | None] = set()
    dims = sorted({str(c.get("dimension") or "") for c in candidates if c.get("dimension")})
    for candidate in candidates:
        sec_key = _normalize_section(candidate.get("section_number"))
        if sec_key in seen_sections:
            continue
        section = section_by_num.get(sec_key)
        if section is None:
            continue
        seen_sections.add(sec_key)
        snippets.append(_section_line(section, 900))

    dim_contexts = [
        f"【{dim} 相关片段】\n{_build_agent_dimension_context(source_text, dim, limit=5000)}"
        for dim in dims
        if dim in _REVIEW_DIMENSIONS
    ]

    context = (
        "【章节标题列表】\n"
        + "\n".join(heading_lines[:240])
        + "\n\n【候选错误所在章节】\n"
        + "\n\n".join(snippets)
        + "\n\n"
        + "\n\n".join(dim_contexts)
    )
    return context[:limit]


def _agent_critic_candidates(
    source_text: str,
    llm,
    candidates: list[dict],
) -> list[dict] | None:
    """Critic Agent：批量反驳和过滤所有 Specialist 候选。"""
    if not candidates:
        return []

    evidence_context = _build_multi_agent_evidence_context(source_text, candidates)
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    system_prompt = """\
你是一名 GB/T 标准审查 Critic Agent。
你的任务不是发现新错误，而是严格反驳和过滤 Specialist Agents 提出的候选错误。

保留候选的条件：
- error_type、dimension、section_number 与证据一致；
- evidence 能支持 reason 和 error_description；
- 不是目录/解析噪声/重复内容造成的误判；
- 不是过度推断或依据不足。

删除候选的条件：
- 证据不足；
- 错误类型不匹配；
- 章节位置不准确；
- 与其他候选重复；
- 只是格式抽取噪声，不构成真实审查错误。

输出 JSON 数组，只保留成立的候选。每个元素必须包含：
error_type、dimension、section_number、reason、error_description、evidence、critic_note。
只输出 JSON 数组，不要输出解释。"""

    user_prompt = f"""\
【文档证据】
{evidence_context}

【Specialist 候选错误】
{candidates_json}

请批量复核并只返回成立的候选错误。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=14000,
    )
    parsed = _parse_llm_json_list(raw, llm)
    if parsed is None:
        return None
    return [item for item in (_normalize_prediction_item(x) for x in parsed) if item is not None]


def _agent_arbiter_decide(
    source_text: str,
    llm,
    candidates: list[dict],
) -> list[dict] | None:
    """Arbiter Agent：最终裁决、合并重复、修正类型/位置。"""
    if not candidates:
        return []

    evidence_context = _build_multi_agent_evidence_context(source_text, candidates)
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    system_prompt = """\
你是一名 GB/T 标准审查 Arbiter Agent。
你需要基于 Critic Agent 保留的候选和文档证据作最终裁决。

要求：
- 合并重复候选；
- 修正明显错误的 error_type、dimension、section_number；
- 删除仍然证据不足或过度推断的候选；
- 保留的每条错误都应能被文档证据直接支持；
- error_description 用“错误：...；修改建议：...；依据：...”格式。

输出 JSON 数组，每个元素必须包含：
error_type、dimension、section_number、reason、error_description、evidence。
只输出 JSON 数组，不要输出解释。"""

    user_prompt = f"""\
【文档证据】
{evidence_context}

【Critic 保留候选】
{candidates_json}

请作最终裁决，返回最终候选错误 JSON 数组。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=14000,
    )
    parsed = _parse_llm_json_list(raw, llm)
    if parsed is None:
        return None
    return [item for item in (_normalize_prediction_item(x) for x in parsed) if item is not None]


def _agent_formatter_output(
    llm,
    candidates: list[dict],
) -> list[dict] | None:
    """Formatter Agent：输出最终标准 predictions JSON，去掉内部字段。"""
    if not candidates:
        return []

    fallback = [
        item
        for item in (_to_prediction_output(candidate) for candidate in _dedupe_predictions(candidates))
        if item is not None
    ]
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    system_prompt = """\
你是一名 JSON Formatter Agent。
请把输入候选错误规范化为最终 predictions JSON 数组。

要求：
- 只保留字段：error_type、dimension、section_number、reason、error_description；
- 删除 evidence、confidence、critic_note 等内部字段；
- 不新增错误；
- 不改变错误含义；
- 确保输出是合法 JSON 数组。

只输出 JSON 数组，不要输出解释。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": candidates_json},
        ],
        temperature=0.0,
        max_tokens=10000,
    )
    parsed = _parse_llm_json_list(raw, llm)
    if parsed is None:
        return fallback

    formatted = [item for item in (_to_prediction_output(x) for x in parsed) if item is not None]
    return _dedupe_predictions(formatted)


def _agent_judge_predictions(
    source_text: str,
    llm,
    predictions: list[dict],
    candidates: list[dict],
    step: int,
    max_steps: int,
) -> dict | None:
    """Judge Agent：判断当前 predictions 是否已经足够好，决定是否提前停止。"""
    predictions_json = json.dumps(predictions, ensure_ascii=False, indent=2)
    candidates_json = json.dumps(candidates[-30:], ensure_ascii=False, indent=2)
    evidence_context = _build_multi_agent_evidence_context(source_text, candidates or predictions, limit=18000)
    system_prompt = """\
你是一名 GB/T 标准审查 Judge Agent。
你的任务是判断当前 multi-agent 审查结果是否已经足够作为最终 predictions。

判断可以停止的条件：
- 当前 predictions 已覆盖文档中证据明确的主要错误；
- error_type、dimension、section_number 基本准确；
- reason 和 error_description 足以支持评测；
- 继续调用 agent 不太可能明显改善结果。

判断需要继续的条件：
- 明显遗漏某些维度；
- 存在较多证据不足、位置不准或类型不准的候选；
- error_description 太弱或不完整；
- 需要某些 Specialist 重新聚焦复查。

输出 JSON 对象：
{
  "is_best": true/false,
  "reason": "一句话说明",
  "feedback": "如果需要继续，给下一轮 agent 的具体反馈；如果可以停止，写最终确认",
  "suggested_dimensions": ["C2.1", "C2.2"]
}

suggested_dimensions 只能从 C2.1、C2.2、C3.1、C3.2、C3.3 中选择；如果不需要继续，返回 []。
只输出 JSON 对象，不要输出解释。"""

    user_prompt = f"""\
【当前步数】
{step}/{max_steps}

【文档证据摘要】
{evidence_context}

【当前 predictions】
{predictions_json}

【最近候选状态】
{candidates_json}

请判断当前 predictions 是否已经达到最佳可用效果。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    parsed = llm.parse_json_response(raw)
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        return {
            "is_best": False,
            "reason": "Judge 未返回对象，继续审查",
            "feedback": "",
            "suggested_dimensions": [],
        }

    suggested = parsed.get("suggested_dimensions") or []
    if not isinstance(suggested, list):
        suggested = []
    parsed["suggested_dimensions"] = [
        dim for dim in suggested
        if dim in _REVIEW_DIMENSIONS
    ]
    parsed["is_best"] = bool(parsed.get("is_best"))
    parsed["feedback"] = str(parsed.get("feedback") or parsed.get("reason") or "").strip()
    parsed["reason"] = str(parsed.get("reason") or "").strip()
    return parsed


def _call_llm_review_agentic_legacy(
    source_text: str,
    llm,
    rag_store: ChromaRAGStore | None = None,
    rag_top_k: int = 5,
    agent_max_steps: int = 12,
) -> list[dict] | None:
    """
    Step-based multi-agent pipeline：
    - 每调用一个 Agent 算一步；
    - Specialist Agent 可并行执行，但每个 Specialist 完成仍计 1 步；
    - 循环在达到 agent_max_steps 或 Judge Agent 判断 predictions 已足够好时退出。
    """
    logger = logging.getLogger(__name__)
    sections = _parse_review_sections(source_text)
    logger.info("[agent] Coordinator 提取章节 %d 个", len(sections))
    if agent_max_steps < 1:
        logger.warning("[agent] agent_max_steps=%d，小于 1，直接返回空结果", agent_max_steps)
        return []

    runnable_dims = list(_REVIEW_DIMENSIONS)
    if not runnable_dims:
        return []

    logger.info("[agent] Specialist-only dimensions: %s", runnable_dims)
    all_candidates: list[dict] = []
    specialist_results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(runnable_dims)) as executor:
        future_to_dim = {
            executor.submit(
                _agent_scan_dimension,
                source_text,
                llm,
                dim,
                rag_store,
                rag_top_k,
                "",
            ): dim
            for dim in runnable_dims
        }

        for future in as_completed(future_to_dim):
            dim = future_to_dim[future]
            try:
                candidates = future.result()
            except Exception as exc:
                logger.error("[agent] Specialist %s failed: %s", dim, exc)
                return None

            if candidates is None:
                logger.warning("[agent] Specialist %s returned null", dim)
                return None

            specialist_results[dim] = candidates
            logger.info("[agent] Specialist %s candidates: %d", dim, len(candidates))

    for dim in runnable_dims:
        all_candidates.extend(specialist_results.get(dim, []))

    deduped_candidates = _dedupe_predictions(all_candidates)
    predictions = [
        item
        for item in (_to_prediction_output(candidate) for candidate in deduped_candidates)
        if item is not None
    ]
    logger.info(
        "[agent] Specialist-only output: raw=%d deduped=%d predictions=%d",
        len(all_candidates),
        len(deduped_candidates),
        len(predictions),
    )
    return predictions

    step = 0
    round_no = 0
    judge_feedback = ""
    pending_dims = list(_REVIEW_DIMENSIONS)
    all_candidates: list[dict] = []
    current_candidates: list[dict] = []
    predictions: list[dict] = []

    while step < agent_max_steps:
        round_no += 1
        dims_to_run = pending_dims or list(_REVIEW_DIMENSIONS)
        remaining_steps = agent_max_steps - step
        runnable_dims = dims_to_run[:remaining_steps]
        if not runnable_dims:
            break

        logger.info(
            "[agent] 第 %d 轮开始，当前步数 %d/%d，Specialist 维度: %s",
            round_no,
            step,
            agent_max_steps,
            runnable_dims,
        )

        specialist_results: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=len(runnable_dims)) as executor:
            future_to_dim = {
                executor.submit(
                    _agent_scan_dimension,
                    source_text,
                    llm,
                    dim,
                    rag_store,
                    rag_top_k,
                    judge_feedback,
                ): dim
                for dim in runnable_dims
            }

            for future in as_completed(future_to_dim):
                dim = future_to_dim[future]
                try:
                    candidates = future.result()
                except Exception as exc:
                    logger.error("[agent] Specialist %s 执行失败: %s", dim, exc)
                    return None

                step += 1
                if candidates is None:
                    logger.warning("[agent] Step %d Specialist %s 返回 null", step, dim)
                    return None

                specialist_results[dim] = candidates
                logger.info(
                    "[agent] Step %d/%d Specialist %s 候选错误 %d 条",
                    step,
                    agent_max_steps,
                    dim,
                    len(candidates),
                )

        for dim in runnable_dims:
            all_candidates.extend(specialist_results.get(dim, []))
        all_candidates = _dedupe_predictions(all_candidates)
        current_candidates = all_candidates
        logger.info("[agent] Specialist 累计候选 %d 条", len(all_candidates))

        if step >= agent_max_steps:
            break
        if not current_candidates:
            if step >= agent_max_steps:
                break
            step += 1
            judge = _agent_judge_predictions(
                source_text,
                llm,
                predictions,
                current_candidates,
                step,
                agent_max_steps,
            )
            if judge is None:
                logger.warning("[agent] Step %d Judge Agent 返回 null", step)
                return None
            logger.info("[agent] Step %d/%d Judge: %s", step, agent_max_steps, judge.get("reason"))
            if judge.get("is_best"):
                return predictions
            judge_feedback = judge.get("feedback") or ""
            pending_dims = judge.get("suggested_dimensions") or list(_REVIEW_DIMENSIONS)
            continue

        step += 1
        criticized = _agent_critic_candidates(source_text, llm, current_candidates)
        if criticized is None:
            logger.warning("[agent] Step %d Critic Agent 返回 null", step)
            return None
        current_candidates = _dedupe_predictions(criticized)
        logger.info(
            "[agent] Step %d/%d Critic Agent 保留 %d 条",
            step,
            agent_max_steps,
            len(current_candidates),
        )
        if step >= agent_max_steps:
            break
        if not current_candidates:
            predictions = []
        else:
            step += 1
            arbitrated = _agent_arbiter_decide(source_text, llm, current_candidates)
            if arbitrated is None:
                logger.warning("[agent] Step %d Arbiter Agent 返回 null", step)
                return None
            current_candidates = _dedupe_predictions(arbitrated)
            logger.info(
                "[agent] Step %d/%d Arbiter Agent 裁决保留 %d 条",
                step,
                agent_max_steps,
                len(current_candidates),
            )

        if step >= agent_max_steps:
            break

        step += 1
        formatted = _agent_formatter_output(llm, current_candidates)
        if formatted is None:
            logger.warning("[agent] Step %d Formatter Agent 返回 null", step)
            return None
        predictions = _dedupe_predictions(formatted)
        logger.info(
            "[agent] Step %d/%d Formatter Agent 输出最终错误 %d 条",
            step,
            agent_max_steps,
            len(predictions),
        )

        if step >= agent_max_steps:
            break

        step += 1
        judge = _agent_judge_predictions(
            source_text,
            llm,
            predictions,
            current_candidates,
            step,
            agent_max_steps,
        )
        if judge is None:
            logger.warning("[agent] Step %d Judge Agent 返回 null", step)
            return None

        logger.info(
            "[agent] Step %d/%d Judge: is_best=%s reason=%s suggested=%s",
            step,
            agent_max_steps,
            judge.get("is_best"),
            judge.get("reason"),
            judge.get("suggested_dimensions"),
        )
        if judge.get("is_best"):
            logger.info("[agent] Judge 判断 predictions 已达到最佳效果，提前退出")
            return predictions

        judge_feedback = judge.get("feedback") or ""
        pending_dims = judge.get("suggested_dimensions") or list(_REVIEW_DIMENSIONS)

    logger.info(
        "[agent] 达到最大步长 %d，返回当前最好 predictions %d 条",
        agent_max_steps,
        len(predictions),
    )
    if predictions:
        return predictions

    fallback = [
        item
        for item in (_to_prediction_output(candidate) for candidate in _dedupe_predictions(current_candidates or all_candidates))
        if item is not None
    ]
    return _dedupe_predictions(fallback)


def _is_valid_prediction_candidate(item: dict) -> bool:
    normalized = _normalize_prediction_item(item)
    if normalized is None:
        return False
    return normalized["error_type"] in _VALID_ERROR_TYPES_BY_DIM.get(normalized["dimension"], set())


def _filter_valid_prediction_candidates(candidates: list[dict]) -> list[dict]:
    return [
        item
        for item in (_normalize_prediction_item(candidate) for candidate in candidates)
        if item is not None and _is_valid_prediction_candidate(item)
    ]


def _mark_candidate_source(candidate: dict, source_agent: str) -> dict:
    item = dict(candidate)
    item["source_agent"] = source_agent
    return item


def _agent_scan_error_type(
    source_text: str,
    llm,
    dim: str,
    error_type: str,
) -> list[dict] | None:
    """Error-Type Agent：只扫描一个低召回错误类型，偏向召回。"""
    if error_type not in _VALID_ERROR_TYPES_BY_DIM.get(dim, set()):
        return []

    focused_context = _build_agent_prefilter_context(source_text, dim)
    guide = _AGENT_DIMENSION_GUIDES.get(dim, "")
    system_prompt = f"""\
你是一名 GB/T 标准审查 Error-Type Agent。
你只负责查找一个错误类型：{dim} / {error_type}。

【当前维度规则】
{guide}

【强制规则】
1. dimension 固定为 "{dim}"。
2. error_type 固定为 "{error_type}"。
3. 目标是最大化召回率：只要某章节可能存在该错误，并且能给出可复核的原文证据，就输出候选。
4. 不要输出其他错误类型；不要把相邻维度的问题混入当前结果。
5. section_number 尽量贴近可能被人工标注为错误位置的章节；跨章节问题可以输出多个候选位置。

输出 JSON 数组。每个元素包含：
error_type、dimension、section_number、reason、error_description、evidence、confidence。
只输出 JSON 数组，不要输出解释。"""

    user_prompt = f"""\
【本地重点上下文】
{focused_context}

【待审查完整正文】
{source_text}

请只查找 {dim} / {error_type}。允许输出中等置信候选，以提高命中率。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=7000,
    )
    parsed = _parse_llm_json_list(raw, llm)
    if parsed is None:
        return None

    items = [item for item in (_normalize_prediction_item(x) for x in parsed) if item is not None]
    return [
        item
        for item in items
        if item["dimension"] == dim and item["error_type"] == error_type
    ]


def _extract_modal_sentence_candidates(source_text: str, max_sentences: int = 220) -> list[dict]:
    """抽取含规范性语气词的句子，供 C3.1 句子级 Agent 使用。"""
    sections = _parse_review_sections(source_text)
    candidates: list[dict] = []
    seen: set[tuple[str | None, str]] = set()
    for section in sections:
        content = str(section.get("content") or "")
        if not any(word in content for word in _MODAL_WORDS):
            continue
        rough_sentences = re.split(r"[。；;！？!?\n]+", content)
        for sentence in rough_sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence) < 6 or not any(word in sentence for word in _MODAL_WORDS):
                continue
            key = (_normalize_section(section.get("section_number")), sentence)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "section_number": section.get("section_number"),
                    "title": section.get("title"),
                    "sentence": sentence[:360],
                    "modal_words": [word for word in _MODAL_WORDS if word in sentence],
                }
            )
            if len(candidates) >= max_sentences:
                return candidates
    return candidates


def _agent_review_c31_sentence_batch(
    llm,
    sentence_candidates: list[dict],
) -> list[dict] | None:
    if not sentence_candidates:
        return []

    candidates_json = json.dumps(sentence_candidates, ensure_ascii=False, indent=2)
    system_prompt = """\
你是一名 GB/T 标准 C3.1 语气审查 Agent。
你将看到一组已经由本地规则抽取出的含语气词句子。

目标是提高召回率：请逐句判断是否可能存在 C3.1 语气类错误。
可以输出中等置信候选；同一句如可能对应多个语气错误类型，可以分别输出。

error_type 只能从以下集合中选择：
E-T-01、E-T-02、E-T-03、E-T-04、E-T-05、E-T-06、E-T-07。

输出 JSON 数组。每个元素必须包含：
error_type、dimension、section_number、reason、error_description、evidence、confidence。
dimension 固定为 "C3.1"。
只输出 JSON 数组，不要输出解释。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": candidates_json},
        ],
        temperature=0.0,
        max_tokens=7000,
    )
    parsed = _parse_llm_json_list(raw, llm)
    if parsed is None:
        return None

    valid_types = _VALID_ERROR_TYPES_BY_DIM["C3.1"]
    items = [item for item in (_normalize_prediction_item(x) for x in parsed) if item is not None]
    return [
        item
        for item in items
        if item["dimension"] == "C3.1" and item["error_type"] in valid_types
    ]


def _agent_review_c31_sentences(
    source_text: str,
    llm,
    batch_size: int = 70,
) -> list[dict] | None:
    sentence_candidates = _extract_modal_sentence_candidates(source_text)
    results: list[dict] = []
    for start in range(0, len(sentence_candidates), batch_size):
        batch = sentence_candidates[start:start + batch_size]
        reviewed = _agent_review_c31_sentence_batch(llm, batch)
        if reviewed is None:
            return None
        results.extend(reviewed)
    return results


def _is_term_section(section: dict) -> bool:
    section_number = str(section.get("section_number") or "")
    title = str(section.get("title") or "")
    return (
        section_number == "3"
        or section_number.startswith("3.")
        or ("术语" in title and "定义" in title)
    )


def _clean_term_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^[\d.]+\s*", "", text)
    text = re.sub(r"\s+[a-zA-Z][a-zA-Z0-9 /,()_-]{1,80}$", "", text).strip()
    return text.strip(" ：:，,。")


def _extract_term_entries(source_text: str, max_terms: int = 120) -> list[dict]:
    sections = _parse_review_sections(source_text)
    entries: list[dict] = []
    seen_terms: set[str] = set()

    for section in sections:
        if not _is_term_section(section):
            continue
        section_number = section.get("section_number")
        title = str(section.get("title") or "")
        content = str(section.get("content") or "")

        if section_number and str(section_number).startswith("3."):
            term = _clean_term_text(title)
            if term and term not in {"术语和定义", "术语", "定义"} and term not in seen_terms:
                seen_terms.add(term)
                entries.append(
                    {
                        "term": term,
                        "section_number": section_number,
                        "definition": _compact_text(content, 260),
                    }
                )

        for match in re.finditer(r"(?:^|\n)\s*(3(?:\.\d+)+)\s+([^\n]{2,100})", content):
            term = _clean_term_text(match.group(2))
            if not term or term in seen_terms:
                continue
            seen_terms.add(term)
            entries.append(
                {
                    "term": term,
                    "section_number": match.group(1),
                    "definition": _compact_text(content[match.end():match.end() + 260], 260),
                }
            )

        if len(entries) >= max_terms:
            break

    return entries[:max_terms]


def _build_c32_term_index_context(source_text: str, max_terms: int = 100) -> tuple[str, list[dict]]:
    sections = _parse_review_sections(source_text)
    term_entries = _extract_term_entries(source_text, max_terms=max_terms)
    body_sections = [section for section in sections if not _is_term_section(section)]

    usage_summary: list[dict] = []
    local_candidates: list[dict] = []
    for entry in term_entries:
        term = str(entry.get("term") or "")
        if len(term) < 2:
            continue
        used_sections = []
        for section in body_sections:
            text = f"{section.get('title')}\n{section.get('content')}"
            if term in text:
                used_sections.append(section.get("section_number"))
        usage_summary.append(
            {
                "term": term,
                "defined_at": entry.get("section_number"),
                "used_sections": used_sections[:12],
                "used_count": len(used_sections),
            }
        )
        if not used_sections and len(local_candidates) < 30:
            local_candidates.append(
                {
                    "error_type": "E-TM-02",
                    "dimension": "C3.2",
                    "section_number": entry.get("section_number"),
                    "reason": f"术语“{term}”在术语和定义中给出，但正文未检出使用位置。",
                    "error_description": f"错误：术语“{term}”在术语和定义章节中定义，但正文未检出使用位置；修改建议：删除未使用术语或在正文相关条款中规范使用；依据：术语和定义应服务于正文技术内容。",
                    "evidence": json.dumps(entry, ensure_ascii=False),
                    "confidence": 0.55,
                }
            )

    term_numbers = []
    for entry in term_entries:
        section_number = str(entry.get("section_number") or "")
        match = re.match(r"^3\.(\d+)$", section_number)
        if match:
            term_numbers.append(int(match.group(1)))
    if term_numbers:
        numbers = sorted(set(term_numbers))
        missing = [num for num in range(numbers[0], numbers[-1] + 1) if num not in numbers]
        if missing:
            local_candidates.append(
                {
                    "error_type": "E-TM-05",
                    "dimension": "C3.2",
                    "section_number": "3",
                    "reason": f"术语编号序列疑似跳号，缺少 {', '.join('3.' + str(num) for num in missing[:8])}。",
                    "error_description": f"错误：术语和定义章节编号不连续，疑似缺少 {', '.join('3.' + str(num) for num in missing[:8])}；修改建议：补齐术语编号或重新连续编号；依据：术语条目编号应保持连续。",
                    "evidence": "术语编号序列：" + ", ".join("3." + str(num) for num in numbers[:80]),
                    "confidence": 0.75,
                }
            )

    context = (
        "【术语表】\n"
        + json.dumps(term_entries, ensure_ascii=False, indent=2)
        + "\n\n【正文使用摘要】\n"
        + json.dumps(usage_summary, ensure_ascii=False, indent=2)
        + "\n\n【正文技术章节片段】\n"
        + _build_c32_technical_element_context(source_text, limit=9000)
    )
    return context[:18000], local_candidates


def _agent_review_c32_term_index(source_text: str, llm) -> list[dict] | None:
    context, local_candidates = _build_c32_term_index_context(source_text)
    system_prompt = """\
你是一名 GB/T 标准 C3.2 术语审查 Agent。
你将看到本地抽取的术语表、术语正文使用摘要和技术章节片段。

目标是提高召回率：请基于术语索引查找 C3.2 术语类错误。
重点关注：
- E-TM-01：正文使用未定义术语；
- E-TM-02：术语定义后正文未使用；
- E-TM-03：正文用法与术语定义不一致；
- E-TM-04：同一概念多术语；
- E-TM-05：术语编号不连续。

允许输出中等置信候选，但必须给出 evidence。
输出 JSON 数组，每个元素包含 error_type、dimension、section_number、reason、error_description、evidence、confidence。
dimension 固定为 "C3.2"。
只输出 JSON 数组，不要输出解释。"""

    raw = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ],
        temperature=0.0,
        max_tokens=9000,
    )
    parsed = _parse_llm_json_list(raw, llm)
    if parsed is None:
        return None
    items = [item for item in (_normalize_prediction_item(x) for x in parsed) if item is not None]
    valid = [
        item
        for item in items
        if item["dimension"] == "C3.2" and item["error_type"] in _VALID_ERROR_TYPES_BY_DIM["C3.2"]
    ]
    return local_candidates + valid


def _reference_base_id(ref: str) -> str:
    ref = _normalize_reference_id(ref)
    return re.sub(r"[-]\d{4}$", "", ref)


def _build_c33_reference_rule_candidates(source_text: str, max_candidates: int = 80) -> list[dict]:
    sections = _parse_review_sections(source_text)
    list_ref_sections: list[dict] = []
    body_ref_sections: list[dict] = []
    for section in sections:
        title = str(section.get("title") or "")
        if "规范性引用文件" in title:
            list_ref_sections.append(section)
        elif "参考文献" not in title:
            text = f"{title}\n{section.get('content') or ''}"
            if _REF_PATTERN.search(text):
                body_ref_sections.append(section)

    list_refs_by_id: dict[str, list[str | None]] = {}
    body_refs_by_id: dict[str, list[str | None]] = {}
    for section in list_ref_sections:
        text = f"{section.get('title')}\n{section.get('content') or ''}"
        for ref in _extract_reference_ids(text):
            list_refs_by_id.setdefault(ref, []).append(section.get("section_number"))
    for section in body_ref_sections:
        text = f"{section.get('title')}\n{section.get('content') or ''}"
        for ref in _extract_reference_ids(text):
            body_refs_by_id.setdefault(ref, []).append(section.get("section_number"))

    list_refs = set(list_refs_by_id)
    body_refs = set(body_refs_by_id)
    candidates: list[dict] = []

    for ref in sorted(body_refs - list_refs):
        section_number = body_refs_by_id.get(ref, [None])[0]
        candidates.append(
            {
                "error_type": "E-R-01",
                "dimension": "C3.3",
                "section_number": section_number,
                "reason": f"正文出现规范性引用“{ref}”，但规范性引用文件清单未检出该条目。",
                "error_description": f"错误：正文引用“{ref}”，但规范性引用文件清单未列出或列出信息不一致；修改建议：补入对应规范性引用文件或修正正文引用；依据：正文规范性引用应在规范性引用文件章列明。",
                "evidence": f"正文引用章节：{body_refs_by_id.get(ref)}；清单引用：{sorted(list_refs)[:40]}",
                "confidence": 0.6,
            }
        )
        if len(candidates) >= max_candidates:
            return candidates

    ref_section_number = list_ref_sections[0].get("section_number") if list_ref_sections else "2"
    for ref in sorted(list_refs - body_refs):
        candidates.append(
            {
                "error_type": "E-R-02",
                "dimension": "C3.3",
                "section_number": ref_section_number,
                "reason": f"规范性引用文件清单列出“{ref}”，但正文未检出对应规范性引用。",
                "error_description": f"错误：规范性引用文件清单列出“{ref}”，但正文未检出对应规范性引用；修改建议：删除冗余引用或在正文中补充规范性引用关系；依据：规范性引用文件应与正文规范性引用相对应。",
                "evidence": f"清单引用章节：{ref_section_number}；正文引用：{sorted(body_refs)[:40]}",
                "confidence": 0.58,
            }
        )
        if len(candidates) >= max_candidates:
            return candidates

    list_by_base: dict[str, set[str]] = {}
    body_by_base: dict[str, set[str]] = {}
    for ref in list_refs:
        list_by_base.setdefault(_reference_base_id(ref), set()).add(ref)
    for ref in body_refs:
        body_by_base.setdefault(_reference_base_id(ref), set()).add(ref)
    for base in sorted(set(list_by_base) & set(body_by_base)):
        if list_by_base[base].isdisjoint(body_by_base[base]):
            body_ref = sorted(body_by_base[base])[0]
            section_number = body_refs_by_id.get(body_ref, [None])[0]
            candidates.append(
                {
                    "error_type": "E-R-03",
                    "dimension": "C3.3",
                    "section_number": section_number,
                    "reason": f"同一引用文件“{base}”在正文和清单中的编号或年代号疑似不一致。",
                    "error_description": f"错误：正文引用与规范性引用文件清单对“{base}”的编号、年代号或版本表述不一致；修改建议：统一正文和清单中的引用文件标识；依据：规范性引用文件信息应前后一致。",
                    "evidence": f"正文：{sorted(body_by_base[base])}；清单：{sorted(list_by_base[base])}",
                    "confidence": 0.52,
                }
            )
        if len(candidates) >= max_candidates:
            return candidates

    return candidates


def _append_reason_suffix(item: dict, suffix: str) -> dict:
    updated = dict(item)
    reason = str(updated.get("reason") or updated.get("error_description") or "").strip()
    if suffix and suffix not in reason:
        updated["reason"] = f"{reason}（{suffix}）" if reason else suffix
    description = str(updated.get("error_description") or updated.get("reason") or "").strip()
    if suffix and suffix not in description:
        updated["error_description"] = f"{description}；补充：{suffix}" if description else suffix
    return updated


def _section_children_by_parent(sections: list[dict]) -> dict[str, list[str]]:
    keys = [
        _normalize_section(section.get("section_number"))
        for section in sections
        if _normalize_section(section.get("section_number")) is not None
    ]
    children: dict[str, list[str]] = {}
    for key in keys:
        if key is None:
            continue
        parts = key.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            children.setdefault(parent, [])
            if key not in children[parent] and key.count(".") == parent.count(".") + 1:
                children[parent].append(key)
    return children


def _expand_section_candidates(
    predictions: list[dict],
    source_text: str,
    max_children: int = 5,
) -> list[dict]:
    sections = _parse_review_sections(source_text)
    section_by_key = {
        _normalize_section(section.get("section_number")): section
        for section in sections
        if _normalize_section(section.get("section_number")) is not None
    }
    children_by_parent = _section_children_by_parent(sections)
    expanded: list[dict] = []

    for prediction in predictions:
        normalized = _normalize_prediction_item(prediction)
        if normalized is None:
            continue
        expanded.append(normalized)
        if normalized["dimension"] not in _SECTION_EXPAND_DIMS:
            continue

        section_key = _normalize_section(normalized.get("section_number"))
        if not section_key:
            continue

        parent_parts = section_key.split(".")
        parent_keys = [".".join(parent_parts[:depth]) for depth in range(len(parent_parts) - 1, 0, -1)]
        for parent_key in parent_keys[:3]:
            candidate = _append_reason_suffix(normalized, f"section 扩展父章节 {parent_key}")
            candidate["section_number"] = section_by_key.get(parent_key, {}).get("section_number", parent_key)
            expanded.append(candidate)

        for child_key in children_by_parent.get(section_key, [])[:max_children]:
            candidate = _append_reason_suffix(normalized, f"section 扩展子章节 {child_key}")
            candidate["section_number"] = section_by_key.get(child_key, {}).get("section_number", child_key)
            expanded.append(candidate)

    return expanded


def _expand_error_type_candidates(predictions: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    for prediction in predictions:
        normalized = _normalize_prediction_item(prediction)
        if normalized is None:
            continue
        expanded.append(normalized)
        expansions = _ERROR_TYPE_EXPANSIONS.get((normalized["dimension"], normalized["error_type"]), [])
        for error_type in expansions:
            if error_type not in _VALID_ERROR_TYPES_BY_DIM.get(normalized["dimension"], set()):
                continue
            candidate = _append_reason_suffix(normalized, f"相近错误类型扩展：{normalized['error_type']} -> {error_type}")
            candidate["error_type"] = error_type
            expanded.append(candidate)
    return expanded


def call_llm_review_agentic(
    source_text: str,
    llm,
    rag_store: ChromaRAGStore | None = None,
    rag_top_k: int = 5,
    agent_max_steps: int = 12,
    use_direct_all: bool = True,
    use_dimension_specialists: bool = True,
    use_error_type_agents: bool = True,
    use_rule_local_scanners: bool = True,
) -> list[dict] | None:
    """
    High-recall multi-agent pipeline:
    1. Run the ALL direct prompt as a broad reviewer.
    2. Run five Dimension Specialists as independent full-scan reviewers.
    3. Run low-recall Error-Type Agents for targeted recall.
    4. Add C3.1 sentence-level, C3.2 term-index, and C3.3 reference-rule candidates.
    5. Expand section_number and nearby error_type candidates, then lightly dedupe.
    """
    logger = logging.getLogger(__name__)
    sections = _parse_review_sections(source_text)
    logger.info("[agent-high-recall] Coordinator 提取章节 %d 个", len(sections))
    logger.info(
        "[agent-high-recall] modules: direct_all=%s dimension_specialists=%s "
        "error_type_agents=%s rule_local_scanners=%s",
        use_direct_all,
        use_dimension_specialists,
        use_error_type_agents,
        use_rule_local_scanners,
    )
    if rag_store is not None:
        logger.info("[agent-high-recall] 当前高召回 agentic 流程不使用 RAG，已忽略 rag_store")

    direct_predictions: list[dict] = []
    if use_direct_all:
        direct_result = call_llm_review(
            source_text,
            llm,
            "ALL",
            rag_store=None,
            rag_top_k=rag_top_k,
        )
        if direct_result is None:
            logger.warning("[agent-high-recall] Direct ALL 返回 null，继续尝试其他召回链路")
            direct_result = []
        direct_predictions = _dedupe_predictions(direct_result)
        logger.info("[agent-high-recall] Direct ALL 预测 %d 条", len(direct_predictions))
    else:
        logger.info("[agent-high-recall] Direct ALL 已关闭")

    if (
        use_direct_all
        and not use_dimension_specialists
        and not use_error_type_agents
        and not use_rule_local_scanners
    ):
        logger.info("[agent-high-recall] Direct ALL Only 消融，仅返回 Direct ALL 结果")
        return [
            item
            for item in (_to_prediction_output(prediction) for prediction in direct_predictions)
            if item is not None
        ]

    if not any([use_direct_all, use_dimension_specialists, use_error_type_agents, use_rule_local_scanners]):
        logger.warning("[agent-high-recall] 所有 agent 模块均关闭，返回空结果")
        direct_predictions = []

    all_candidates: list[dict] = [
        _mark_candidate_source(prediction, "direct_all")
        for prediction in direct_predictions
    ]

    if agent_max_steps < 1:
        logger.info("[agent-high-recall] agent_max_steps=%d，仅返回 Direct ALL 结果", agent_max_steps)
        return [
            item
            for item in (_to_prediction_output(prediction) for prediction in direct_predictions)
            if item is not None
        ]

    if use_dimension_specialists:
        runnable_dims = list(_REVIEW_DIMENSIONS)
        max_workers = min(len(runnable_dims), max(1, agent_max_steps))
        logger.info("[agent-high-recall] Dimension Specialists 全量扫描: %s, max_workers=%d", runnable_dims, max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dim = {
                executor.submit(
                    _agent_scan_dimension,
                    source_text,
                    llm,
                    dim,
                    None,
                    rag_top_k,
                    "最大化召回率；允许中等置信候选；同一章节可输出多个可能 error_type。",
                    None,
                    "full_scan",
                ): dim
                for dim in runnable_dims
            }

            for future in as_completed(future_to_dim):
                dim = future_to_dim[future]
                try:
                    candidates = future.result()
                except Exception as exc:
                    logger.warning("[agent-high-recall] Specialist %s 全量扫描失败，继续其他链路: %s", dim, exc)
                    continue

                if candidates is None:
                    logger.warning("[agent-high-recall] Specialist %s 返回 null，跳过该维度", dim)
                    continue

                logger.info("[agent-high-recall] Specialist %s 候选 %d 条", dim, len(candidates))
                all_candidates.extend(_mark_candidate_source(candidate, f"specialist_{dim}") for candidate in candidates)
    else:
        logger.info("[agent-high-recall] Dimension Specialists 已关闭")

    if use_error_type_agents:
        error_type_workers = min(len(_HIGH_RECALL_ERROR_TYPE_AGENTS), max(1, agent_max_steps))
        logger.info(
            "[agent-high-recall] Error-Type Agents 专项扫描: %s, max_workers=%d",
            _HIGH_RECALL_ERROR_TYPE_AGENTS,
            error_type_workers,
        )
        with ThreadPoolExecutor(max_workers=error_type_workers) as executor:
            future_to_error = {
                executor.submit(_agent_scan_error_type, source_text, llm, dim, error_type): (dim, error_type)
                for dim, error_type in _HIGH_RECALL_ERROR_TYPE_AGENTS
            }
            for future in as_completed(future_to_error):
                dim, error_type = future_to_error[future]
                try:
                    candidates = future.result()
                except Exception as exc:
                    logger.warning("[agent-high-recall] Error-Type Agent %s/%s 失败: %s", dim, error_type, exc)
                    continue
                if candidates is None:
                    logger.warning("[agent-high-recall] Error-Type Agent %s/%s 返回 null", dim, error_type)
                    continue
                logger.info("[agent-high-recall] Error-Type Agent %s/%s 候选 %d 条", dim, error_type, len(candidates))
                all_candidates.extend(
                    _mark_candidate_source(candidate, f"error_type_{error_type}")
                    for candidate in candidates
                )
    else:
        logger.info("[agent-high-recall] Error-Type Agents 已关闭")

    if use_rule_local_scanners:
        try:
            c31_sentence_candidates = _agent_review_c31_sentences(source_text, llm)
            if c31_sentence_candidates is not None:
                logger.info("[agent-high-recall] C3.1 句子级扫描候选 %d 条", len(c31_sentence_candidates))
                all_candidates.extend(
                    _mark_candidate_source(candidate, "c31_sentence_scan")
                    for candidate in c31_sentence_candidates
                )
            else:
                logger.warning("[agent-high-recall] C3.1 句子级扫描返回 null")
        except Exception as exc:
            logger.warning("[agent-high-recall] C3.1 句子级扫描失败: %s", exc)

        try:
            c32_term_candidates = _agent_review_c32_term_index(source_text, llm)
            if c32_term_candidates is not None:
                logger.info("[agent-high-recall] C3.2 术语索引扫描候选 %d 条", len(c32_term_candidates))
                all_candidates.extend(
                    _mark_candidate_source(candidate, "c32_term_index")
                    for candidate in c32_term_candidates
                )
            else:
                logger.warning("[agent-high-recall] C3.2 术语索引扫描返回 null")
        except Exception as exc:
            logger.warning("[agent-high-recall] C3.2 术语索引扫描失败: %s", exc)

        try:
            c33_rule_candidates = _build_c33_reference_rule_candidates(source_text)
            logger.info("[agent-high-recall] C3.3 引用规则扫描候选 %d 条", len(c33_rule_candidates))
            all_candidates.extend(
                _mark_candidate_source(candidate, "c33_reference_rules")
                for candidate in c33_rule_candidates
            )
        except Exception as exc:
            logger.warning("[agent-high-recall] C3.3 引用规则扫描失败: %s", exc)
    else:
        logger.info("[agent-high-recall] Rule/Local Scanners 已关闭")

    valid_candidates = _filter_valid_prediction_candidates(all_candidates)
    section_expanded = _expand_section_candidates(valid_candidates, source_text)
    type_expanded = _expand_error_type_candidates(section_expanded)
    merged = _dedupe_predictions(_filter_valid_prediction_candidates(type_expanded))
    predictions = [
        item
        for item in (_to_prediction_output(prediction) for prediction in merged)
        if item is not None
    ]
    logger.info(
        "[agent-high-recall] final predictions=%d (raw=%d, valid=%d, section_expanded=%d, type_expanded=%d)",
        len(predictions),
        len(all_candidates),
        len(valid_candidates),
        len(section_expanded),
        len(type_expanded),
    )
    return predictions


def build_ground_truth(doc: dict) -> list[dict]:
    """从文档的 examples 提取 ground truth 列表。"""
    ground_truth: list[dict] = []
    for ex in doc.get("examples", []):
        gt_section = ex.get("corrupted") or ex.get("original") or {}
        ground_truth.append(
            {
                "error_type":        ex["error_type"],
                "dimension":         ex["dimension"],
                # "section_number":    ex["original"]["section_number"],#corrupted
                "section_number":    gt_section.get("section_number"),
                "title":             gt_section.get("title"),
                "content":           gt_section.get("content", ""),
                "error_description": ex.get("error_description", ""),
            }
        )
    return ground_truth


def _section_display_number(section: dict | None, fallback=None):
    if section is not None:
        return section.get("section_number")
    return fallback


def _build_marked_hit_sections_text(hit_sections: list[dict]) -> str:
    blocks: list[str] = []
    for section in hit_sections:
        error_types = ",".join(section.get("hit_error_types") or [])
        dimensions = ",".join(section.get("hit_dimensions") or [])
        marker = (
            f">>> HIT level={section.get('hit_level')} "
            f"section={section.get('section_number')} "
            f"dimension={dimensions} error_type={error_types}"
        )
        title = section.get("title") or ""
        number = section.get("section_number")
        heading = f"[{number}] {title}" if number is not None else title
        content = section.get("content") or ""
        blocks.append(f"{marker}\n{heading}\n{content}".rstrip())
    return "\n\n".join(blocks)


def build_prediction_section_marks(
    source_text: str,
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """Mark each prediction and collect source sections hit by predictions."""
    sections = _parse_review_sections(source_text)
    section_by_key: dict[str | None, dict] = {}
    section_order: dict[str | None, int] = {}
    for index, section in enumerate(sections):
        key = _normalize_section(section.get("section_number"))
        if key not in section_by_key:
            section_by_key[key] = section
            section_order[key] = index

    prediction_matches: list[dict] = []
    hit_section_map: dict[tuple[str | None, str | None, str | None, int], dict] = {}

    for pred_index, pred in enumerate(predictions):
        pred_sec = _normalize_section(pred.get("section_number"))
        location_matches = [
            gt_index
            for gt_index, gt in enumerate(ground_truth)
            if _normalize_section(gt.get("section_number")) == pred_sec
        ]
        diagnosis_matches = [
            gt_index
            for gt_index in location_matches
            if pred.get("dimension") == ground_truth[gt_index].get("dimension")
            and pred.get("error_type") == ground_truth[gt_index].get("error_type")
        ]
        matched_gt_index = (
            diagnosis_matches[0]
            if diagnosis_matches
            else (location_matches[0] if location_matches else None)
        )
        section = section_by_key.get(pred_sec)
        prediction_mark = {
            "prediction_index": pred_index,
            "prediction": pred,
            "location_hit": bool(location_matches),
            "diagnosis_hit": bool(diagnosis_matches),
            "matched_gt_index": matched_gt_index,
            "matched_gt": ground_truth[matched_gt_index] if matched_gt_index is not None else None,
            "matched_section_number": _section_display_number(section, pred.get("section_number")),
            "matched_section_title": section.get("title") if section is not None else None,
        }
        prediction_matches.append(prediction_mark)

        if not diagnosis_matches:
            continue

        for gt_index in diagnosis_matches:
            gt = ground_truth[gt_index]
            normalized_section = _normalize_section(gt.get("section_number"))
            gt_section = section_by_key.get(normalized_section)
            section_key = (
                normalized_section,
                gt.get("dimension"),
                gt.get("error_type"),
                gt_index,
            )

            title = gt.get("title")
            content = gt.get("content")
            if title is None and gt_section is not None:
                title = gt_section.get("title")
            if not content and gt_section is not None:
                content = gt_section.get("content") or ""

            if section_key not in hit_section_map:
                hit_section_map[section_key] = {
                    "section_number": gt.get("section_number"),
                    "normalized_section_number": normalized_section,
                    "title": title,
                    "content": content or "",
                    "hit_level": "diagnosis",
                    "prediction_indices": [],
                    "matched_gt_indices": [gt_index],
                    "hit_error_types": [gt.get("error_type")] if gt.get("error_type") else [],
                    "hit_dimensions": [gt.get("dimension")] if gt.get("dimension") else [],
                    "section_order": section_order.get(normalized_section, 10**9),
                }

            hit_section = hit_section_map[section_key]
            hit_section["prediction_indices"].append(pred_index)

    hit_sections = sorted(
        hit_section_map.values(),
        key=lambda item: (item.get("section_order", 10**9), str(item.get("section_number") or "")),
    )
    for section in hit_sections:
        section.pop("section_order", None)

    return {
        "prediction_matches": prediction_matches,
        "hit_sections": hit_sections,
        "marked_hit_sections_text": _build_marked_hit_sections_text(hit_sections),
    }


def build_eval_record(doc: dict, predictions: list[dict], ground_truth: list[dict]) -> dict:
    """将单篇文档的 ground truth 与预测结果合并为一条评测记录。"""
    marks = build_prediction_section_marks(
        doc.get("source_text", ""),
        predictions,
        ground_truth,
    )
    return {
        "file_name":    doc["file_name"],
        "file_stem":    doc["file_stem"],
        "ground_truth": ground_truth,
        "predictions":  predictions,
        **marks,
    }


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GB/T 反例审查评测脚本")
    parser.add_argument(
        "--input", "-i",
        default="data/GBT_Data_fanli_10to17/GBT_test_balanced_00.json",
        help="输入 GBT_test.json 路径（默认使用仓库内第一个 benchmark shard）",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/eval_result.json",
        help="评测结果输出路径（默认 outputs/eval_result.json）",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["proxy", "azure"],
        default="proxy",
        help="LLM 后端：proxy 或 azure（默认 proxy）",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="仅处理前 N 篇文档（调试用，默认全部）",
    )
    parser.add_argument(
        "--dim", "-d",
        default="ALL",
        choices=["ALL", *_REVIEW_DIMENSIONS],
        help=f"审查维度；当前统一使用 ALL prompt，传入单维度会自动按 ALL 审查（默认 ALL，审查 {_REVIEW_DIMENSIONS}）",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="启用本地 ChromaDB + Ollama embedding RAG 审查增强",
    )
    parser.add_argument(
        "--rag-db",
        default=RAG_CONFIG["persist_dir"],
        help=f"ChromaDB 持久化目录（默认 {RAG_CONFIG['persist_dir']}）",
    )
    parser.add_argument(
        "--rag-seed",
        default=RAG_CONFIG["seed_path"],
        help=f"RAG 初始知识库 JSONL（默认 {RAG_CONFIG['seed_path']}）",
    )
    parser.add_argument(
        "--rag-collection",
        default=RAG_CONFIG["collection_name"],
        help=f"Chroma collection 名称（默认 {RAG_CONFIG['collection_name']}）",
    )
    parser.add_argument(
        "--rag-embedding-model",
        default=RAG_CONFIG["embedding_model"],
        help=f"Ollama embedding 模型（默认 {RAG_CONFIG['embedding_model']}）",
    )
    parser.add_argument(
        "--rag-ollama-url",
        default=RAG_CONFIG["ollama_base_url"],
        help=f"Ollama 服务地址（默认 {RAG_CONFIG['ollama_base_url']}）",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=RAG_CONFIG["top_k"],
        help=f"每篇文档检索的知识块数量（默认 {RAG_CONFIG['top_k']}）",
    )
    parser.add_argument(
        "--agentIsTrue",
        action="store_true",
        help="启用高召回 multi-agent pipeline；不传则使用原有单次 prompt 审查",
    )
    parser.add_argument(
        "--agentMaxSteps",
        type=int,
        default=12,
        help="高召回 multi-agent 并发上限；小于 1 时仅运行 Direct ALL，默认 12",
    )
    parser.add_argument(
        "--agent-direct-all",
        "--agentDirectAll",
        dest="agent_direct_all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Agent 消融开关：启用/关闭 Direct ALL Review（默认启用）",
    )
    parser.add_argument(
        "--agent-dimension-specialists",
        "--agentDimensionSpecialists",
        dest="agent_dimension_specialists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Agent 消融开关：启用/关闭 Dimension Specialists（默认启用）",
    )
    parser.add_argument(
        "--agent-error-type-agents",
        "--agentErrorTypeAgents",
        dest="agent_error_type_agents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Agent 消融开关：启用/关闭 Error-Type Agents（默认启用）",
    )
    parser.add_argument(
        "--agent-rule-local-scanners",
        "--agentRuleLocalScanners",
        dest="agent_rule_local_scanners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Agent 消融开关：启用/关闭 Rule/Local Scanners（默认启用）",
    )
    args = parser.parse_args()
    if args.dim in _REVIEW_DIMENSIONS:
        args.dim = "ALL"
    return args


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    setup_logging(level=LOG_CONFIG["level"], log_file=LOG_CONFIG["file"])
    logger = logging.getLogger(__name__)
    logger.info("=== GBT 反例审查评测启动 ===")
    logger.info("输入文件: %s", args.input)
    agentIsTrue = bool(args.agentIsTrue)
    logger.info("审查模式: %s", "agentic pipeline" if agentIsTrue else "single prompt")
    agent_modules = {
        "direct_all": bool(args.agent_direct_all),
        "dimension_specialists": bool(args.agent_dimension_specialists),
        "error_type_agents": bool(args.agent_error_type_agents),
        "rule_local_scanners": bool(args.agent_rule_local_scanners),
    }
    if agentIsTrue:
        logger.info("Agent 模块开关: %s", agent_modules)

    # ── 初始化 LLM ────────────────────────────────────────────────
    if args.backend == "azure":
        logger.info("使用 Azure OpenAI 后端")
        llm = AzureLLMClient(AZURE_LLM_CONFIG)
    else:
        logger.info("使用代理 OpenAI 后端")
        llm = OpenAILLMClient(LLM_CONFIG)

    # ── 初始化 RAG（可选）──────────────────────────────────────────
    rag_store = None
    if args.use_rag:
        logger.info(
            "启用 RAG: chroma=%s collection=%s embedding=%s ollama=%s",
            args.rag_db,
            args.rag_collection,
            args.rag_embedding_model,
            args.rag_ollama_url,
        )
        rag_store = ChromaRAGStore(
            persist_dir=args.rag_db,
            collection_name=args.rag_collection,
            embedding_model=args.rag_embedding_model,
            ollama_base_url=args.rag_ollama_url,
            seed_path=args.rag_seed,
        )

    # ── 读取测试数据 ───────────────────────────────────────────────
    data = load_test_json(args.input)
    documents = data.get("documents", [])
    if args.limit:
        documents = documents[: args.limit]
    total = len(documents)
    logger.info("共加载 %d 篇文档", total)

    # ── 准备输出文件（流式写入） ────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    # 文档级计数器
    g_precision_hit = 0   # 检测正例文章数
    g_location_hit  = 0   # 定位正例文章数
    g_diagnosis_hit = 0   # 诊断正例文章数
    g_h_hits = {f"H{k}": 0 for k in range(1, 11)}
    g_item_recall = Counter()
    g_item_recall_by_dimension: dict[str, Counter] = {}
    g_item_recall_by_error_type: dict[str, Counter] = {}
    g_item_recall_error_type_meta: dict[str, dict] = {}

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f'{{\n  "total": {total},\n  "records": [\n')

        for idx, doc in enumerate(documents, 1):
            stem = doc.get("file_stem", f"doc_{idx}")
            logger.info("── 审查 [%d/%d]: %s ──", idx, total, doc.get("file_name", stem))

            try:
                # ① 取出正文
                source_text = doc.get("source_text", "")
                if not source_text.strip():
                    logger.warning("[%s] source_text 为空，跳过", stem)
                    continue

                # ② 调用 LLM 审查
                if agentIsTrue:
                    predictions = call_llm_review_agentic(
                        source_text,
                        llm,
                        rag_store=rag_store,
                        rag_top_k=args.rag_top_k,
                        agent_max_steps=args.agentMaxSteps,
                        use_direct_all=args.agent_direct_all,
                        use_dimension_specialists=args.agent_dimension_specialists,
                        use_error_type_agents=args.agent_error_type_agents,
                        use_rule_local_scanners=args.agent_rule_local_scanners,
                    )
                else:
                    predictions = call_llm_review(
                        source_text,
                        llm,
                        args.dim,
                        rag_store=rag_store,
                        rag_top_k=args.rag_top_k,
                    )
                if predictions is None:
                    logger.warning("[%s] 模型返回 predictions=null，跳过该文档且不计入准确率", stem)
                    continue
                logger.info("[%s] 模型预测 %d 处错误", stem, len(predictions))

                # ③ 提取 ground truth
                ground_truth = build_ground_truth(doc)

                # ④ 文档级二值判定
                p_hit, l_hit, d_hit = judge_doc(ground_truth, predictions, llm=llm)
                h_matched_count, h_metrics = judge_diagnosis_hit_counts(ground_truth, predictions)
                item_recall = build_item_recall_metrics(ground_truth, predictions)
                g_precision_hit += int(p_hit)
                g_location_hit  += int(l_hit)
                g_diagnosis_hit += int(d_hit)
                for key, hit in h_metrics.items():
                    g_h_hits[key] += int(hit)
                _accumulate_item_recall(g_item_recall, item_recall)
                update_item_recall_group_totals(
                    g_item_recall_by_dimension,
                    ground_truth,
                    predictions,
                    "dimension",
                )
                update_item_recall_group_totals(
                    g_item_recall_by_error_type,
                    ground_truth,
                    predictions,
                    "error_type",
                    metadata=g_item_recall_error_type_meta,
                )
                logger.info(
                    "[%s] Precision=%s  Location=%s  Diagnosis=%s  DiagnosisMatches=%d",
                    stem,
                    "✓" if p_hit else "✗",
                    "✓" if l_hit else "✗",
                    "✓" if d_hit else "✗",
                    h_matched_count,
                )

                # ⑤ 构造评测记录并流式写入
                record = build_eval_record(doc, predictions, ground_truth)
                record["review_mode"] = "agentic" if agentIsTrue else "single_prompt"
                if agentIsTrue:
                    record["agent_max_steps"] = args.agentMaxSteps
                    record["agent_modules"] = agent_modules
                record["doc_metrics"] = {
                    "precision_hit": p_hit,
                    "location_hit":  l_hit,
                    "diagnosis_hit": d_hit,
                    "diagnosis_matched_count": h_matched_count,
                    **h_metrics,
                }
                record["item_recall"] = item_recall
                if written > 0:
                    f.write(",\n")
                record_str = json.dumps(record, ensure_ascii=False, indent=2)
                indented = "\n".join("    " + line for line in record_str.splitlines())
                f.write(indented)
                f.flush()
                written += 1

            except Exception as exc:
                logger.error("[%d/%d] 处理失败 %s: %s", idx, total, stem, exc)

        f.write("\n  ]\n}\n")

    # ── 计算并输出全局三项指标 ─────────────────────────────────────
    precision     = g_precision_hit / written if written > 0 else 0.0
    location_acc  = g_location_hit  / written if written > 0 else 0.0
    diagnosis_acc = g_diagnosis_hit / written if written > 0 else 0.0
    h_rates = {key: value / written if written > 0 else 0.0 for key, value in g_h_hits.items()}
    item_recall_summary = _finalize_item_recall(g_item_recall)
    item_recall_summary["by_dimension"] = finalize_item_recall_groups(
        g_item_recall_by_dimension,
    )
    item_recall_summary["by_error_type"] = finalize_item_recall_groups(
        g_item_recall_by_error_type,
        metadata=g_item_recall_error_type_meta,
    )

    summary = {
        "review_mode":        "agentic" if agentIsTrue else "single_prompt",
        **({"agent_max_steps": args.agentMaxSteps} if agentIsTrue else {}),
        **({"agent_modules": agent_modules} if agentIsTrue else {}),
        "doc_count":          written,
        "precision_hit":      g_precision_hit,
        "location_hit":       g_location_hit,
        "diagnosis_hit":      g_diagnosis_hit,
        "precision":          round(precision,     4),
        "location_accuracy":  round(location_acc,  4),
        "diagnosis_accuracy": round(diagnosis_acc, 4),
        "diagnosis_H_hits":   g_h_hits,
        "diagnosis_H":        {key: round(value, 4) for key, value in h_rates.items()},
        "item_recall":        item_recall_summary,
        "np_mcs_params": {
            "formula": "doc_weighted_np_mcs = weighted_mean(MCS_doc * (1 - alpha * FP_proxy_doc / (FP_proxy_doc + beta)), weight=GT_doc)",
            "mcs_doc": "diagnosis_recall_doc * exp(-lambda * miss_count_doc / ground_truth_count_doc)",
            "false_positive_proxy_doc": "max(prediction_count_doc - diagnosis_hit_doc, 0)",
            "lambda": NP_MCS_LAMBDA,
            "alpha": NP_MCS_ALPHA,
            "beta": NP_MCS_BETA,
        },
    }

    # 打印到终端
    print("\n" + "=" * 60)
    print(f"  评测完成  共 {written} 篇文档")
    print(f"  诊断层  Precision          : {precision:.4f}  ({g_precision_hit}/{written})")
    print(f"  检测层  Diagnosis Accuracy : {diagnosis_acc:.4f}  ({g_diagnosis_hit}/{written})")
    print(f"  定位层  Location Accuracy  : {location_acc:.4f}  ({g_location_hit}/{written})")
    print(
        "  条目级  Diagnosis Recall   : "
        f"{item_recall_summary['diagnosis_recall']:.4f}  "
        f"({item_recall_summary['diagnosis_hit']}/{item_recall_summary['ground_truth_count']})"
    )
    print(
        "  条目级  Soft NP-MCS        : "
        f"{item_recall_summary.get('doc_weighted_np_mcs_soft_alpha015_beta100', 0.0):.4f}"
    )
    for key in sorted(g_h_hits, key=lambda item: int(item[1:])):
        print(f"  检测层  {key:<3}                 : {h_rates[key]:.4f}  ({g_h_hits[key]}/{written})")
    print("=" * 60)

    # 保存到 summary 文件
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("汇总指标已写入: %s", summary_path)
    logger.info("=== 评测完成，共处理 %d/%d 篇，结果写入: %s ===", written, total, output_path)


if __name__ == "__main__":
    main()
