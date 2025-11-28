import os
import re


def read_schema(schema_path):
    '''
    Read and clean the .schema file into a plain text description.
    This will be prepended to the prompt for LLM-based text-to-SQL.
    '''
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    with open(schema_path, "r") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        l = line.strip()
        if not l or l.startswith("--") or l.startswith("#"):
            continue
        cleaned.append(l)

    schema_text = "\n".join(cleaned)
    return schema_text


def extract_sql_query(response):
    '''
    Extract the SQL query from a model's text output.
    We handle various possible patterns the model might produce.
    '''
    if not response:
        return ""

    # Common pattern: if model outputs something like "SQL: SELECT ..."
    match = re.search(r"(SELECT .*?);?\s*$", response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
    else:
        lines = response.splitlines()
        sql = ""
        for line in lines:
            if "select" in line.lower():
                sql = line.strip()
                break

    # Remove markdown or code fences
    sql = sql.replace("```sql", "").replace("```", "").strip()

    if sql and not sql.endswith(";"):
        sql += ";"

    return sql


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to a .txt file.
    '''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em:.4f}\n")
        f.write(f"Record EM: {record_em:.4f}\n")
        f.write(f"Record F1: {record_f1:.4f}\n")
        f.write(f"Error Messages: {error_msgs}\n")

    print(f"[LOG] Saved experiment log to {output_path}")
