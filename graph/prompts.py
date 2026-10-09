"""All LLM prompts and fixed reply texts used by the LangGraph nodes."""

from langchain_core.prompts import ChatPromptTemplate

NO_ANSWER = "I don't know based on the provided project repository."
OUT_OF_SCOPE_ANSWER = (
    "This question is not related to the project repository, so I can't answer it. "
    "I can only answer questions about the projects in our knowledge base, such as their "
    "technologies, team size, duration, domain, solution approach or outcomes."
)

REPOSITORY_DESCRIPTION = (
    "The repository describes about 100 projects. Each project has: Project ID (e.g. P038), "
    "Project Name, Type (ML/AI, NLP / GenAI or Data Engineering), Domain, Team Size, Duration, "
    "Technology Stack, Problem Statement, Solution Approach, Expected Outcomes and "
    "Key Metrics & Business Impact."
)

# Used by analyze_query: rewrites the question for retrieval and extracts its conditions.
QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze a user's question about a project repository.\n"
     f"{REPOSITORY_DESCRIPTION}\n\n"
     "Return two things:\n"
     "1. search_query: one line for a hybrid retriever (semantic FAISS + keyword BM25). "
     "Keep every project ID, project name, technology, number, unit and condition. Add the "
     "relevant field/section names (e.g. 'technology stack', 'team size', 'duration', "
     "'solution approach'). Fix obvious spelling mistakes. Do not answer the question.\n"
     "2. The explicit conditions the projects must satisfy (team_size, duration, "
     "project_types, technologies).\n\n"
     "Rules for conditions:\n"
     "- Users make typos and use informal wording: understand the meaning. "
     "'atleast 5' = >= 5, 'more then 3 months' = > 3 months, '5+ members' = >= 5, "
     "'max 4 people' = <= 4, 'a quarter' = 3 months, 'half a year' = 6 months.\n"
     "- Only fill a condition the question actually states. Never invent conditions.\n"
     "- Leave all conditions empty if the question is about specific project IDs (e.g. P038).\n"
     "- project_types: only when the question restricts the category "
     "(e.g. 'GenAI projects' -> NLP / GenAI, 'machine learning projects' -> ML/AI).\n"
     "- technologies: only specific named tools, languages or platforms (e.g. Python, "
     "Snowflake, Kafka), with spelling corrected. Not generic words like 'AI' or 'cloud'.\n"
     "- If the question is unrelated to the repository (e.g. weather, news), set search_query "
     "to the question as written and leave every condition null."),
    ("human", "{question}"),
])

# Used by refine_query (retry only): writes a better search query from the verifier feedback.
REFINE_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "A previous search did not retrieve enough context to answer a question correctly. "
     "Write a better search query for a hybrid retriever (semantic FAISS + keyword BM25).\n"
     f"{REPOSITORY_DESCRIPTION}\n\n"
     "Rules:\n"
     "- Keep every project ID, project name, technology, number, unit and comparison from the question.\n"
     "- Use the verifier feedback to target the missing information.\n"
     "- Do not answer the question.\n"
     "- Return only the new search query on a single line."),
    ("human",
     "Question: {question}\n"
     "Previous search query: {search_query}\n"
     "Verifier feedback: {feedback}"),
])

# Used by generate_answer: writes the answer from the retrieved context only.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a project repository using ONLY the context provided.\n"
     f"{REPOSITORY_DESCRIPTION}\n\n"
     "Rules:\n"
     "- First decide whether the QUESTION is about this project repository (its projects, "
     "technologies, teams, domains, durations, problems, solutions or outcomes). If it is about "
     "anything else (weather, news, sports, general knowledge, coding help, chit-chat, ...), do "
     f"not answer it and reply exactly: \"{OUT_OF_SCOPE_ANSWER}\" This applies even if the "
     "context happens to contain related words.\n"
     "- Use only the retrieved context. Do not use outside knowledge.\n"
     "- Do not invent projects or project information.\n"
     "- Copy project IDs, project names, team sizes, durations and technologies exactly "
     "as they appear in the context.\n"
     "- When listing a project's technology stack, include every technology listed for it.\n"
     "- When the question asks about one section (e.g. solution approach, problem statement, "
     "outcomes), report what THAT section says, keeping its key points and exact numbers "
     "(e.g. '10,000+', '120K'). Do not blend in facts from other sections (e.g. do not add "
     "'runs on AWS' because AWS is in the technology stack) and do not add connecting claims "
     "that the text does not state.\n"
     "- If a 'PROJECTS MATCHING THE METADATA FILTERS' list is given, it was computed "
     "deterministically from the structured project database and is complete for those "
     "conditions. For filtering/listing questions, include every project in that list and "
     "no other projects. Do not re-check the numeric conditions yourself.\n"
     "- For other filtering or comparison questions, evaluate all relevant projects in the "
     "context and return every supported match.\n"
     f"- If the context does not contain enough information, reply exactly: \"{NO_ANSWER}\"\n"
     "- Never mention a project called UNKNOWN.\n"
     "- Be concise and readable. Use bullet points for lists, never repeat an item, and cite "
     "projects by their project ID only (no markers like [Source 1])."),
    ("human",
     "{feedback_note}"
     "CONTEXT:\n{context}\n\n"
     "QUESTION: {question}"),
])

# Appended to QUERY_ANALYSIS_PROMPT / VERIFICATION_PROMPT only when strict JSON-schema
# output fails and the call is repeated in plain JSON mode.
JSON_FALLBACK_INSTRUCTION = (
    "Respond with ONLY a JSON object (no reasoning, no other text) that matches this JSON schema:\n"
    "{json_schema}"
)

# Used by verify_answer: fact-checks the answer and returns a verdict, not a new answer.
VERIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker for a RAG system. Do NOT write a new answer. "
     "Check the ANSWER against the CONTEXT and the QUESTION:\n"
     "1. Is every statement grounded in the context?\n"
     "2. Are project IDs correct?\n"
     "3. Are project names correct?\n"
     "4. Are team sizes correct?\n"
     "5. Are durations correct?\n"
     "6. Are technologies correct, with none missing from a listed technology stack?\n"
     "7. Is anything hallucinated? Compare claims against the specific section they describe: "
     "a statement presented as part of a project's solution approach (or problem, outcomes) "
     "must be stated in that section. Facts borrowed from another section (e.g. 'runs on AWS' "
     "taken from the technology stack), changed numbers, or invented details count as "
     "hallucinations.\n"
     "8. For filtering/listing questions, were all relevant candidates considered? If a "
     "'PROJECTS MATCHING THE METADATA FILTERS' list is given, the answer must include exactly "
     "those projects.\n"
     "9. Does the answer actually answer the question?\n"
     f"The reply \"{NO_ANSWER}\" is valid only if the context truly lacks the information.\n"
     f"The reply \"{OUT_OF_SCOPE_ANSWER}\" is valid only if the question is NOT about the "
     "project repository; if the question IS about the repository, that reply is invalid.\n"
     "Set is_valid to true only if all checks pass. In feedback, briefly name the specific "
     "problems, or write 'All checks passed.'"),
    ("human",
     "CONTEXT:\n{context}\n\n"
     "QUESTION: {question}\n\n"
     "ANSWER:\n{answer}"),
])
