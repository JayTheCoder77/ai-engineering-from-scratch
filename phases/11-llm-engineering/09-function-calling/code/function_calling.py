import json
import math
import time


TOOL_REGISTRY = {}
TOOL_CACHE={}
CACHE_HITS=0
CACHE_MISSES=0

def make_hashable(val):
    if isinstance(val , dict):
        return frozenset((k , make_hashable(v)) for k , v in val.items())
    elif isinstance(val , (list, tuple)):
        return tuple(make_hashable(v) for v in val)
    else:
        return val

def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }


def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(
            code,
            {
                "__builtins__": {
                    "print": print, "range": range, "len": len, "str": str,
                    "int": int, "float": float, "list": list, "dict": dict,
                    "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
                    "sorted": sorted, "enumerate": enumerate, "zip": zip,
                    "map": map, "filter": filter, "math": math,
                }
            },
            local_vars,
        )
        result = local_vars.get("result", None)
        return {
            "success": True,
            "result": result,
            "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")},
        }
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}

# 6th tool -> database query tool

DB_TABLES = {
    "users": [
            {"id": 1, "name": "Alice", "age": 30, "role": "admin"},
            {"id": 2, "name": "Bob", "age": 25, "role": "user"},
            {"id": 3, "name": "Charlie", "age": 35, "role": "user"},
            {"id": 4, "name": "Diana", "age": 28, "role": "admin"},
        ],
    "products": [
        {"id": 1, "name": "Laptop", "price": 1200, "stock": 10},
        {"id": 2, "name": "Phone", "price": 800, "stock": 25},
        {"id": 3, "name": "Tablet", "price": 450, "stock": 0},
        {"id": 4, "name": "Monitor", "price": 300, "stock": 15},
    ]
}

def query_database(table_name , filters=None):
    if table_name not in DB_TABLES:
        return {
            "error" : True,
            "message" : f"Table {table_name} not found",
            "allowed_tables" : list(DB_TABLES.keys()),
            "code" : "TABLE_NOT_FOUND"
        }

    rows = DB_TABLES[table_name]
    if not filters:
        return {"table": table_name, "rows": rows, "count": len(rows)}
    
    filtered_rows = []
    allowed_operators = {"=", ">", "<", ">=", "<="}

    for row in rows:
        match = True
        for f in filters:
            col = f.get("column")
            op = f.get("operator")
            val = f.get("value")

            # Check if column exists in table rows
            if col not in row:
                return {
                    "error": True,
                    "message": f"Column '{col}' does not exist in table '{table_name}'.",
                    "code": "COLUMN_NOT_FOUND"
                }
                    
            # Validate operator
            if op not in allowed_operators:
                return {
                    "error": True,
                    "message": f"Invalid operator '{op}'. Allowed operators are: = , > , < , >= , <=.",
                    "code": "INVALID_OPERATOR"
                }

            # Perform comparison
            row_val = row[col]
            try:
                if op == "=":
                    if row_val != val:
                        match = False
                elif op == ">":
                    if not (row_val > val):
                        match = False
                elif op == "<":
                    if not (row_val < val):
                        match = False
                elif op == ">=":
                    if not (row_val >= val):
                        match = False
                elif op == "<=":
                    if not (row_val <= val):
                        match = False
            except TypeError as e:
                # Handle type mismatches (e.g. comparing string to number)
                return {
                    "error": True,
                    "message": f"Type mismatch comparing column '{col}' of type {type(row_val).__name__} with value {val} of type {type(val).__name__}.",
                    "code": "TYPE_MISMATCH"
                }
                
            if not match:
                break
                
        if match:
            filtered_rows.append(row)
            
    return {"table": table_name, "rows": filtered_rows, "count":
len(filtered_rows)}


def register_all_tools():
    register_tool(
        "calculator",
        "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"},
                "precision": {"type": "integer", "description": "Decimal places in result", "default": 2},
            },
            "required": ["expression"],
        },
        calculator,
    )
    register_tool(
        "get_weather",
        "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"},
            },
            "required": ["city"],
        },
        get_weather,
    )
    register_tool(
        "web_search",
        "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3},
            },
            "required": ["query"],
        },
        web_search,
    )
    register_tool(
        "read_file",
        "Read the contents of a file. Returns the file content, size, and line count.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"},
            },
            "required": ["path"],
        },
        read_file,
    )
    register_tool(
        "run_code",
        "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "language": {"type": "string", "enum": ["python"], "description": "Programming language"},
            },
            "required": ["code"],
        },
        run_code,
    )
    register_tool(
            "query_database",
            "Query an in-memory database table with filters. Returns matching rows as JSON.",
            {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "enum" : ["users" , "products"],
                        "description": "Name of the table and Allowed values"
                    },
                    "filters": {
                        "type": "array",
                        "description": "Optional list of filter conditions to apply (AND logic).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "description": "Column name to filter on"},
                                "operator": {"type": "string", "enum": ["=", ">", "<",
  ">=", "<="], "description": "Comparison operator"},
                                "value": {"type": ["string", "number"],
  "description": "Value to compare against"}
                            },
                            "required": ["column", "operator", "value"]
                        }
                    }
                },
                "required": ["table_name"],
            },
            query_database,
        )

def simulate_model_decision(user_message, tools, conversation_history):
    # Check if the last action was a tool execution that failed
    if conversation_history:
        last_msg = conversation_history[-1]
        if last_msg.get("role") == "tool":
            try:
                result_content = json.loads(last_msg.get("content", "{}"))
                if isinstance(result_content, dict) and result_content.get("error"):
                    tool_name = last_msg.get("tool_name")
                    error_msg = result_content.get("message", "")
                    
                    # Correction 1: Weather suggestions
                    if tool_name == "get_weather" and "suggestions" in result_content:
                        suggestions = result_content.get("suggestions", [])
                        if suggestions:
                            return [{"name": "get_weather", "arguments": {"city": suggestions[0].title()}}]
                            
                    # Correction 2: Calculator syntax retry
                    if tool_name == "calculator" and "invalid syntax" in error_msg:
                        # Fallback: Extract the entire math expression from the user's initial prompt
                        user_prompt = conversation_history[0].get("content", "")
                        expr = "".join(c for c in user_prompt if c in "0123456789+-*/.() ")
                        if expr.strip():
                            return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
            except json.JSONDecodeError:
                pass

    msg = user_message.lower()

    if conversation_history and not ("config" in msg and ("price" in msg or "pricing" in msg)):
        last_msg = conversation_history[-1]
        if last_msg.get("role") == "tool":
            try:
                result_content = json.loads(last_msg.get("content", "{}"))
                if isinstance(result_content, dict) and not result_content.get("error"):
                    return []
            except json.JSONDecodeError:
                pass
    
    if "config" in msg and ("price" in msg or "pricing" in msg):
        read_result = None
        search_result = None

        for hist_msg in conversation_history:
            if hist_msg.get("role") == "tool":
                if hist_msg.get("tool_name") == "read_file":
                    read_result = hist_msg.get("content")
                elif hist_msg.get("tool_name") == "web_search":
                    search_result = hist_msg.get("content")
        
        if not read_result:
            return [{"name" : "read_file" , "arguments" : {"path" : "data/config.json"}}]
        elif not search_result:
            try:
                data = json.loads(read_result)
                file_content = json.loads(data.get("content"," {}"))
                
                model_name = file_content.get("model" , "gpt-4o")
            except Exception:
                model_name = "gpt-4o"
            return [{"name" : "web_search" , "arguments" : {"query" : f"{model_name} pricing"}}]

        else:
            return []

    print(f"Conversation history {conversation_history}")
    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            import re
            match = re.search(r'in\s+([a-zA-Z]+)', msg)
            if match:
                cities = [match.group(1)]
            else:
                cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the    sandbox!'", "language": "python"}}]
    
    if any(word in msg for word in ["query", "database", "table", "users",
  "products"]):
            # A simple parser for demo queries
            table = "users" if "user" in msg else "products"
            filters = []
            if "older than" in msg or "age >" in msg:
                # extract number
                import re
                match = re.search(r'(?:older than|age >)\s*(\d+)', msg)
                if match:
                    filters.append({"column": "age", "operator": ">", "value":
  int(match.group(1))})
            elif "admin" in msg:
                filters.append({"column": "role", "operator": "=", "value": "admin"})
            # ... we can expand parser logic as needed
            return [{"name": "query_database", "arguments": {"table_name": table, "filters": filters}}]
    # if none of the above match, return an empty list of tool calls
    return []


def execute_tool_call(tool_call):
    global CACHE_HITS, CACHE_MISSES
    name = tool_call["name"]
    args = tool_call["arguments"]
    cache_key = (name , make_hashable(args))

    if cache_key in TOOL_CACHE:
        cached_time , cached_result = TOOL_CACHE[cache_key]

        if time.time() - cached_time < 60 :
            CACHE_HITS += 1
            return {"tool" : name , "result" : cached_result , "execution_time_ms" : 0}

    if name not in TOOL_REGISTRY:
        return {"tool": name, "result": {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}, "execution_time_ms": 0}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)

    TOOL_CACHE[cache_key] = (time.time() , result)
    CACHE_MISSES += 1
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}

def evaluate_caching():
        global CACHE_HITS, CACHE_MISSES
        # Reset cache and statistics
        TOOL_CACHE.clear()
        CACHE_HITS = 0
        CACHE_MISSES = 0
        
        conversation_queries = [
            "What's the weather in Tokyo?",
            "What's the weather in London?",
            "What's the weather in Tokyo?",
            "Calculate (10 + 5) * 3",
            "Calculate (10 + 5) * 3",
            "Read the config file and tell me what model is configured,then search the web for that model's pricing.",
            "Read the config file and tell me what model is configured, then search the web for that model's pricing.",
            "Search for python function calling",
            "Search for python function calling",
            "Show me all admin users",
            "Show me all admin users",
            "Calculate 100 / 4",
            "Calculate 100 / 4",
            "What's the weather in London?",
            "Search for MCP protocol",
            "Search for MCP protocol",
            "Show me all admin users",
            "What's the weather in Tokyo?",
            "Calculate (10 + 5) * 3",
            "Read the config file and tell me what model is configured,then search the web for that model's pricing."
        ]
        

        print(f"\n===========================================================")
        print(f"  Tool Call Caching Evaluation (20 Queries)")

        print(f"=============================================================")
        
        for i, query in enumerate(conversation_queries):
            print(f"\n  Query {i+1}: '{query}'")
            run_function_calling_loop(query)
            print(f"    Current Stats -> Hits: {CACHE_HITS} | Misses: {CACHE_MISSES}")
            
        total_calls = CACHE_HITS + CACHE_MISSES
        hit_rate = (CACHE_HITS / total_calls) * 100 if total_calls > 0 else 0
        
        print(f"\n-------------------------------------------------------------")
        print(f"Caching Evaluation Summary:")
        print(f"  Total executed tool calls: {total_calls}")
        print(f"  Cache Hits:                {CACHE_HITS}")
        print(f"  Cache Misses:              {CACHE_MISSES}")
        print(f"  Cache Hit Rate:            {hit_rate:.2f}%")
        print(f"-------------------------------------------------------------")

def validate_value(value, schema, path="", errors=None):
    if errors is None:
        errors = []
        
    expected_type = schema.get("type")
    
    # 1. Type validation
    if expected_type:
        type_checks = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        # Check if type is a list of choices (like ["string", "number"])
        if isinstance(expected_type, list):
            valid = False
            for t in expected_type:
                if t in type_checks and isinstance(value, type_checks[t]):
                    valid = True
                    break
            if not valid:
                errors.append(f"Argument '{path}': expected one of {expected_type}, got {type(value).__name__}")
                return errors
        elif expected_type in type_checks:
            if not isinstance(value, type_checks[expected_type]):
                errors.append(f"Argument '{path}': expected {expected_type}, got {type(value).__name__}")
                return errors

    # 2. Enum validation
    if "enum" in schema:
        if value not in schema["enum"]:
            errors.append(f"Argument '{path}': '{value}' not in {schema['enum']}")

    # 3. Recursive checks for arrays and objects
    if expected_type == "array" and isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate_value(item, schema["items"], f"{path}[{i}]", errors)
            
    elif expected_type == "object" and isinstance(value, dict):
        required = schema.get("required", [])
        for req_field in required:
            if req_field not in value:
                field_path = f"{path}.{req_field}" if path else req_field
                errors.append(f"Missing required argument: {field_path}")
        
        properties = schema.get("properties", {})
        for prop_name, prop_value in value.items():
            if prop_name not in properties:
                field_path = f"{path}.{prop_name}" if path else prop_name
                errors.append(f"Unknown argument: {field_path}")
                continue
            
            sub_path = f"{path}.{prop_name}" if path else prop_name
            validate_value(prop_value, properties[prop_name], sub_path, errors)
            
    return errors

def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    return validate_value(arguments, schema)
# def validate_tool_arguments(tool_name, arguments):
#     if tool_name not in TOOL_REGISTRY:
#         return [f"Unknown tool: {tool_name}"]

#     schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
#     errors = []

#     if not isinstance(arguments, dict):
#         return [f"Arguments must be an object, got {type(arguments).__name__}"]

#     for required_field in schema.get("required", []):
#         if required_field not in arguments:
#             errors.append(f"Missing required argument: {required_field}")

#     properties = schema.get("properties", {})
#     for arg_name, arg_value in arguments.items():
#         if arg_name not in properties:
#             errors.append(f"Unknown argument: {arg_name}")
#             continue

#         prop_schema = properties[arg_name]
#         expected_type = prop_schema.get("type")

#         type_checks = {
#             "string": str,
#             "integer": int,
#             "number": (int, float),
#             "boolean": bool,
#             "array": list,
#             "object": dict,
#         }
#         if expected_type in type_checks:
#             if not isinstance(arg_value, type_checks[expected_type]):
#                 errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

#         if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
#             errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

#     return errors


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []
    retry_counts = {}  # Unique call key -> retry count

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            # Create a unique key for this tool + arguments combo
            call_key = (call["name"], json.dumps(call["arguments"], sort_keys=True))
            
            # Check if this call has exceeded max retries
            if retry_counts.get(call_key, 0) >= 3:
                print(f"      [Retry limit reached for {call['name']} with args {call['arguments']}]")
                results.append({
                    "tool": call["name"],
                    "result": {"error": True, "message": "Max retries (3) reached for this tool call.", "code": "MAX_RETRIES_EXCEEDED"},
                    "execution_time_ms": 0
                })
                continue
            
            result = execute_tool_call(call)
            results.append(result)
            
            # If the tool call returned an error, increment retry count
            if isinstance(result["result"], dict) and result["result"].get("error"):
                retry_counts[call_key] = retry_counts.get(call_key, 0) + 1
                print(f"      [Error detected in {call['name']}. Retry count: {retry_counts[call_key]}/3]")

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({
                "role": "tool",
                "content": json.dumps(result["result"]),
                "tool_name": result["tool"],
            })

        all_tool_results.extend(results)
        
        # Check if we should continue the loop
        has_error = any(isinstance(r["result"], dict) and r["result"].get("error") for r in results)
        
        # # If there are no errors, we are done
        # if not has_error:
        #     break
        if has_error:
            # If all errors in this turn are due to reaching the retry limit, break to prevent infinite loops
            all_errors_exceeded = all(
                isinstance(r["result"], dict) and r["result"].get("code") == "MAX_RETRIES_EXCEEDED"
                for r in results if isinstance(r["result"] , dict) and r["result"].get("error")
            )
            if all_errors_exceeded:
                break
            

    return {
        "conversation": conversation,
        "tool_results": all_tool_results,
        # "iterations": iteration + 1 if tool_calls else 0,
        "iterations": iteration,
        "retry_counts": {f"{k[0]}({k[1]})": v for k, v in retry_counts.items()}
    }

def evaluate_accuracy():
    queries_dataset = [
        # (Query, Expected Tool)
        ("Is it raining in London?", "get_weather"),
        ("Temperature in San Francisco please", "get_weather"),
        ("Check the forecast for Sydney", "get_weather"),
        ("What is the weather like in New York?", "get_weather"),
        ("Do I need an umbrella in Tokyo today?", "get_weather"),
        ("Compute (45 + 55) * 2", "calculator"),
        ("What is 1024 divided by 4?", "calculator"),
        ("Calculate 15.5 * 3", "calculator"),
        ("How much is 99 - 45?", "calculator"),
        ("Add 12 and 34", "calculator"),
        ("Search for python function calling guides", "web_search"),
        ("Google MCP protocol spec", "web_search"),
        ("Find information on weather API", "web_search"),
        ("Look up open-source LLMs", "web_search"),
        ("Search details about DeepSeek-V3", "web_search"),
        ("Read file README.md", "read_file"),
        ("Show me the config JSON file content", "read_file"),
        ("Cat users CSV data", "read_file"),
        ("Open data/config.json", "read_file"),
        ("Read data/users.csv", "read_file"),
        ("Run code to calculate fibonacci", "run_code"),
        ("Execute python script for sorting list", "run_code"),
        ("Run some Python instructions", "run_code"),
        ("Show me all admin users in the database", "query_database"),
        ("Query products with price > 500", "query_database"),
        ("Get user table data from database", "query_database"),
        ("Query products table for Laptop", "query_database"),
        ("Hello, how are you today?", None),  # Direct text - no tool
        ("Can you tell me a funny joke?", None),  # Direct text - no tool
        ("Explain function calling in simple words", None)  # Direct text - no tool
    ]

    correct_count = 0
    total_count = len(queries_dataset)
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    mismatches = []


    for query, expected_tool in queries_dataset:
        tool_calls = simulate_model_decision(query , tool_definitions, [] )
        actual_tool = tool_calls[0]["name"] if tool_calls else None
        if actual_tool == expected_tool:
            correct_count += 1
        else:
            mismatches.append((query , expected_tool, actual_tool))
        
    accuracy = (correct_count / total_count) * 100
    print(f"\n============================================================")
    print(f"  Tool Selection Accuracy Evaluation")
    print(f"============================================================")
    print(f"Total Queries: {total_count}")
    print(f"Correct Selections: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
        
    if mismatches:
        print("\n--- Mismatch / Confusion Report ---")
        for query, expected, actual in mismatches:
            print(f"  Query: '{query}'")
            print(f"    Expected: {expected}")
            print(f"    Got:      {actual}\n")
    else:
        print("\nPerfect selection! No confusion detected.")
    

def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
        ("query_database", {"table_name": "users", "filters": [{"column": "age", "operator": ">=", "value": 30}]}, "Valid DB query"),
        ("query_database", {"table_name": "orders"}, "Invalid table (orders is not in allowlist)"),
        ("query_database", {"table_name": "users", "filters": [{"column": "age", "operator": "like", "value": "30"}]}, "Invalid operator in filter")
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
        {"name": "query_database", "arguments": {"table_name": "users", "filters": [{"column": "role", "operator": "=", "value": "admin"}]}},
        {"name": "query_database", "arguments": {"table_name": "products", "filters": [{"column": "price", "operator": ">", "value": 500}]}}
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "What's the weather in Toky?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file and tell me what model is configured, then search the web for that model's pricing.",
        "Run some Python code",
        "Tell me a joke",
        "Show me all admin users",
        "Query products with price more than 500"
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")

    print(f"\n--- Accuracy Evaluation ---")
    evaluate_accuracy()

    print(f"\n--- Tool Call Caching Evaluation ---")
    
    evaluate_caching()

    print(f"\n--- Full Demo Completed ---")


if __name__ == "__main__":
    run_demo()
