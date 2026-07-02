BASE_INPUT_RATE = 0.015
CACHE_WRITE_RATE = 0.01875
CACHE_READ_RATE = 0.0015

SYSTEM_PROMPT_TOKENS = 5000
USER_TOKENS_PER_TURN = 200
ASSISTANT_TOKENS_PER_TURN = 300
NUM_TURNS = 10

total_tokens_no_cache = 0
total_cost_no_cache = 0.0

for turn in range(1, NUM_TURNS + 1):
    accumulated_history = (turn - 1) * (
        USER_TOKENS_PER_TURN + ASSISTANT_TOKENS_PER_TURN
    )
    turn_input_tokens = (
        SYSTEM_PROMPT_TOKENS + accumulated_history + USER_TOKENS_PER_TURN
    )
    turn_cost = (turn_input_tokens / 1000) * BASE_INPUT_RATE
    total_tokens_no_cache += turn_input_tokens
    total_cost_no_cache += turn_cost

    print(
        f"Without Cache: Total Tokens = {total_tokens_no_cache}, Total Cost = ${total_cost_no_cache:.5f}"
    )


# System Prompt Cached Only
total_cost_cache = 0.0

for turn in range(1, NUM_TURNS + 1):
    # Calculate the history + current user message tokens (dynamic part)
    accumulated_history = (turn - 1) * (
        USER_TOKENS_PER_TURN + ASSISTANT_TOKENS_PER_TURN
    )
    dynamic_tokens = accumulated_history + USER_TOKENS_PER_TURN

    # Calculate the dynamic part cost
    dynamic_cost = (dynamic_tokens / 1000) * BASE_INPUT_RATE

    # Determine the system prompt cost depending on whether it's Turn 1 (Write) or Turn 2+ (Read)
    if turn == 1:
        # Write to cache
        system_cost = (SYSTEM_PROMPT_TOKENS / 1000) * CACHE_WRITE_RATE
    else:
        # Read from cache
        system_cost = (SYSTEM_PROMPT_TOKENS / 1000) * CACHE_READ_RATE

    total_cost_cache += system_cost + dynamic_cost
    print(f"With Cache: Total Cost = ${total_cost_cache:.5f}")

total_cost_cache_history = 0.0

for turn in range(1, NUM_TURNS + 1):

    user_msg_cost = (USER_TOKENS_PER_TURN / 1000) * BASE_INPUT_RATE

    if turn == 1:
        # Write to cache
        system_cost = (SYSTEM_PROMPT_TOKENS / 1000) * CACHE_WRITE_RATE
        total_cost_cache_history += system_cost + user_msg_cost
    else:
        # Read from cache
        warm_tokens = SYSTEM_PROMPT_TOKENS + (turn - 2) * (
            USER_TOKENS_PER_TURN + ASSISTANT_TOKENS_PER_TURN
        )
        read_cost = (warm_tokens / 1000) * CACHE_READ_RATE

        new_history_tokens = USER_TOKENS_PER_TURN + ASSISTANT_TOKENS_PER_TURN
        write_cost = (new_history_tokens / 1000) * CACHE_WRITE_RATE

        total_cost_cache_history += read_cost + write_cost + user_msg_cost

    print(f"With History Cache: Total Cost = ${total_cost_cache_history:.5f}")
