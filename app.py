import streamlit as st
from loguru import logger

from migo.config import DEFAULT_MAX_ROUNDS
from migo.nodes.evaluator import evaluator
from migo.nodes.knowledge_map_updater import knowledge_map_updater
from migo.nodes.question_generator import question_generator
from migo.nodes.router import router

st.set_page_config(page_title="Migo", page_icon="🧠")
st.title("Migo")
st.caption("Adaptive trivia �� powered by Wikipedia + Claude")

# --- Session state init ---
if "game_state" not in st.session_state:
    st.session_state.game_state = None
    st.session_state.round_active = False
    st.session_state.index = None
    st.session_state.phase = "idle"  # idle | question | answered | summary
    st.session_state.history = []

# --- Topic selection ---
topic = st.text_input("Enter a topic to quiz on", placeholder="e.g. World War 2")

if st.button("Start Round", disabled=not topic):
    category = "general_knowledge"
    concept = topic.strip().lower().replace(" ", "_")

    with st.spinner("Fetching Wikipedia article and building index..."):
        try:
            from migo.retrieval.indexer import index_topic, query_topic

            index = index_topic(topic.strip())
            st.session_state.index = index

            chunks = query_topic(index, concept)
            retrieved_chunks = [c.text for c in chunks]
            source_url = chunks[0].source_url if chunks else ""
            logger.info("Retrieved {} chunks for '{}'", len(chunks), concept)

        except (ValueError, ConnectionError) as e:
            st.error(f"Could not fetch topic: {e}")
            st.stop()

    st.session_state.game_state = {
        "user_id": "default",
        "category": category,
        "current_topic": concept,
        "current_question": "",
        "current_options": [],
        "correct_answer": "",
        "user_answer": "",
        "is_correct": None,
        "explanation": "",
        "source_url": source_url,
        "retrieved_chunks": retrieved_chunks,
        "knowledge_map": {
            category: {
                concept: {
                    "confidence": 0.0,
                    "attempts": 0,
                    "streak": 0,
                    "last_tested": None,
                },
            },
        },
        "round_number": 0,
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "round_score": 0,
        "round_history": [],
        "should_end": False,
    }
    st.session_state.round_active = True
    st.session_state.phase = "question"
    st.session_state.history = []

    # Generate first question via router + question_generator
    gs = st.session_state.game_state
    gs.update(router(gs))
    if not gs.get("should_end"):
        with st.spinner("Generating question..."):
            gs.update(question_generator(gs))
    st.rerun()

# --- Game loop ---
if st.session_state.round_active and st.session_state.game_state:
    gs = st.session_state.game_state
    phase = st.session_state.phase

    st.markdown(f"**Round {gs['round_number']}/{gs['max_rounds']}** | Score: **{gs['round_score']}**")
    st.divider()

    # -- Question phase --
    if phase == "question" and gs["current_question"]:
        st.markdown(f"### Q{gs['round_number']}: {gs['current_question']}")

        if gs["current_options"]:
            answer = st.radio("Pick your answer:", gs["current_options"], index=None)
        else:
            answer = st.text_input("Your answer:")

        if st.button("Submit Answer", disabled=not answer):
            gs["user_answer"] = answer
            with st.spinner("Evaluating..."):
                gs.update(evaluator(gs))
                gs.update(knowledge_map_updater(gs))

            st.session_state.history.append({
                "question": gs["current_question"],
                "user_answer": gs["user_answer"],
                "correct_answer": gs["correct_answer"],
                "is_correct": gs["is_correct"],
                "explanation": gs["explanation"],
                "source_url": gs["source_url"],
            })
            st.session_state.phase = "answered"
            st.rerun()

    # -- Answered phase --
    elif phase == "answered":
        last = st.session_state.history[-1]
        st.markdown(f"### Q{gs['round_number']}: {last['question']}")
        st.markdown(f"**Your answer:** {last['user_answer']}")

        if last["is_correct"]:
            st.success("Correct!")
        else:
            st.error(f"Incorrect. The answer was: {last['correct_answer']}")

        st.info(last["explanation"])

        if last["source_url"]:
            st.markdown(f"[Wikipedia source]({last['source_url']})")

        if st.button("Next Question"):
            gs.update(router(gs))

            if gs.get("should_end"):
                st.session_state.phase = "summary"
                st.rerun()
            else:
                with st.spinner("Generating question..."):
                    gs.update(question_generator(gs))
                st.session_state.phase = "question"
                st.rerun()

    # -- Summary phase --
    elif phase == "summary":
        st.markdown("---")
        st.markdown(f"## Round Complete!  Score: {gs['round_score']}/{gs['max_rounds']}")

        for i, h in enumerate(st.session_state.history, 1):
            icon = "✅" if h["is_correct"] else "❌"
            st.markdown(f"**{icon} Q{i}:** {h['question']}")
            st.markdown(f"  Your answer: {h['user_answer']}")
            if not h["is_correct"]:
                st.markdown(f"  Correct answer: {h['correct_answer']}")
            st.markdown(f"  {h['explanation']}")
            if h["source_url"]:
                st.markdown(f"  [Source]({h['source_url']})")
            st.markdown("")

        if st.button("Play Again"):
            st.session_state.game_state = None
            st.session_state.round_active = False
            st.session_state.phase = "idle"
            st.session_state.history = []
            st.rerun()
