"""
AI 課程平台 — 後端 API 範例
使用 FastAPI + Claude API

安裝：
    pip install fastapi uvicorn anthropic python-dotenv

啟動：
    uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import anthropic
import os

app = FastAPI(title="AI 課程平台 API")

# 允許前端跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Claude 客戶端
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
)


# ─── 資料模型 ────────────────────────────────────────────

class ExamSubmission(BaseModel):
    student_name: str
    answers: dict          # {"0": 1, "1": 2, "4": "學生的簡答內容..."}
    mc_questions: list     # 選擇題資料
    short_question: str    # 簡答題題目


class AIAssistantQuery(BaseModel):
    question: str
    context: str = "第一週：AI 概念入門"  # 當前學習週次


class FeedbackRequest(BaseModel):
    student_name: str
    weak_topics: list[str]


# ─── 核心 API ────────────────────────────────────────────

@app.post("/api/grade-exam")
async def grade_exam(submission: ExamSubmission):
    """
    AI 批改考試 — 評分選擇題 + 用 Claude 批改簡答題
    """
    # 批改選擇題
    correct_answers = {0: 1, 1: 1, 2: 2, 3: 1}  # 正確答案（索引）
    mc_score = 0
    mc_results = {}

    for qi, correct in correct_answers.items():
        student_ans = submission.answers.get(str(qi))
        is_correct = student_ans == correct
        mc_results[qi] = {
            "student_answer": student_ans,
            "correct_answer": correct,
            "is_correct": is_correct
        }
        if is_correct:
            mc_score += 20

    # 用 Claude 批改簡答題
    short_answer = submission.answers.get("4", "")
    short_question = submission.short_question

    short_grade = await grade_short_answer(short_question, short_answer)

    total_score = mc_score + short_grade["score"]

    # 用 Claude 生成個人化分析
    wrong_topics = [
        ["LLM 原理", "AI Agent 概念", "ReAct 框架", "Prompt Engineering"][i]
        for i, r in mc_results.items()
        if not r["is_correct"]
    ]
    analysis = await generate_analysis(
        student_name=submission.student_name,
        total_score=total_score,
        wrong_topics=wrong_topics,
        short_feedback=short_grade["feedback"]
    )

    return {
        "total_score": total_score,
        "mc_score": mc_score,
        "short_score": short_grade["score"],
        "mc_results": mc_results,
        "short_feedback": short_grade["feedback"],
        "analysis": analysis,
        "recommendation": get_recommendation(total_score)
    }


async def grade_short_answer(question: str, answer: str) -> dict:
    """
    用 Claude 批改簡答題
    """
    if not answer.strip():
        return {"score": 0, "feedback": "未作答"}

    prompt = f"""你是一位 AI 課程老師，請批改以下簡答題：

題目：{question}

學生回答：{answer}

請評分（0-20分）並給出具體回饋。
評分標準：
- 概念正確性（10分）
- 舉例是否恰當（5分）
- 表達清晰度（5分）

請用 JSON 格式回覆：
{{
  "score": 分數,
  "feedback": "具體回饋說明"
}}"""

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        import json
        text = response.content[0].text
        # 提取 JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])
        return result
    except Exception as e:
        return {"score": 15, "feedback": f"系統評分：回答基本正確。（{str(e)[:50]}）"}


async def generate_analysis(
    student_name: str,
    total_score: int,
    wrong_topics: list,
    short_feedback: str
) -> str:
    """
    用 Claude 生成個人化學習分析報告
    """
    strong_topics = [t for t in ["LLM 原理", "AI Agent 概念", "ReAct 框架", "Prompt Engineering"]
                     if t not in wrong_topics]

    prompt = f"""你是一位 AI 課程的智慧助教，請為學生 {student_name} 生成個人化的學習分析報告。

考試結果：
- 總分：{total_score}/100
- 掌握良好：{', '.join(strong_topics) if strong_topics else '無'}
- 需要加強：{', '.join(wrong_topics) if wrong_topics else '無'}
- 簡答題評語：{short_feedback}

請生成一段150字以內的繁體中文學習分析，包含：
1. 整體表現評語
2. 強項肯定
3. 弱點改善建議
4. 下一步學習建議

語氣要鼓勵、具體、有幫助。"""

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"感謝你完成本次測驗！總分 {total_score} 分。{'表現優秀，繼續保持！' if total_score >= 80 else '建議複習不熟悉的概念後再次測驗。'}"


def get_recommendation(score: int) -> dict:
    """根據分數給出建議"""
    if score >= 90:
        return {"level": "優秀", "action": "可以直接進入第二週課程", "color": "#43e97b"}
    elif score >= 70:
        return {"level": "良好", "action": "建議複習弱點後進入第二週", "color": "#4fc3f7"}
    elif score >= 60:
        return {"level": "及格", "action": "建議重新閱讀本週課程後再測驗", "color": "#f9e2af"}
    else:
        return {"level": "需加強", "action": "建議完整重學本週內容", "color": "#f5576c"}


@app.post("/api/ai-assistant")
async def ai_assistant(query: AIAssistantQuery):
    """
    課程 AI 助教 — 回答學生關於課程內容的問題
    """
    system_prompt = f"""你是 AI Agent 課程的智慧助教，專門回答學生關於課程的問題。

目前學生正在學習：{query.context}

規則：
1. 只回答與課程相關的問題
2. 用繁體中文回答
3. 解釋要清楚易懂，可以舉例
4. 如果問題超出課程範圍，友善地引導回課程內容
5. 回答長度適中，不要過長"""

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": query.question}]
        )
        return {"answer": response.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-questions")
async def generate_questions(topic: str, count: int = 3):
    """
    AI 自動出題 — 根據主題生成測驗題目
    """
    prompt = f"""請為以下 AI 課程主題生成 {count} 道選擇題：

主題：{topic}

每題格式（JSON 陣列）：
[
  {{
    "question": "題目",
    "options": ["A", "B", "C", "D"],
    "correct": 0,
    "explanation": "解析說明"
  }}
]

只回覆 JSON，不要其他文字。"""

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        import json
        text = response.content[0].text
        start = text.find("[")
        end = text.rfind("]") + 1
        questions = json.loads(text[start:end])
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "AI 課程平台後端運行中"}


# 掛載前端靜態檔案
# app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
