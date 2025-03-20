import json
import re
import sqlglot

def extract_sql_part(query, db_type):
    try:
        sqlglot.parse(sql, dialect=db_type)
        return query
    except:
        pass

    init_tokens = ['select', 'with']
    i_codeblock = query.find('```')
    query = query.strip()
    last_semicolon = False

    if i_codeblock != -1:
        query = query[i_codeblock:]

    queryl = query.lower()

    if not query.lower().startswith('with'):
        i_start = None

        for init_token in init_tokens:
            i_start = queryl.find(init_token)
            if i_start != -1:
                break
            if not i_start:
                return None

        query = query[i_start:]

    if query.endswith(';'):
        query = query.rstrip(';')
        last_semicolon = True

    i_semicolon = query.rfind(';')

    if i_semicolon != -1:
        last_part = query[i_semicolon + 1 :].lower()

        if all(token not in last_part for token in init_tokens):
            query = query[: i_semicolon + 1]

    if last_semicolon and not query.endswith(';'):
        query += ';'

    return query

def fix_invalid_col_in_select(ast, col_table_map, invalid_ident, db_type):
    selects = ast.find_all(sqlglot.expressions.Select)
    eq_pairs = find_eqs(ast)
    aliases = find_aliases(ast)
    updated = False

    for select in selects:
        select_cols = []

        for expr in select.args['expressions']:
            if isinstance(expr, sqlglot.expressions.Column):
                select_cols.append(expr)

        from_info = select.args.get("from")

        if from_info is None:
            continue
        else:
            from_expr = from_info.this

        if isinstance(from_expr, sqlglot.expressions.Table):
            table = from_expr.name
        else:
            continue

        where_expr = select.args.get("where")

        if not where_expr:
            continue

        where_eqs = find_eqs(where_expr)

        for where_eq in where_eqs:
            col = where_eq[0]
            val = where_eq[1]

            if col.sql().upper() in aliases:
                continue

            if col.sql().upper() in col_table_map and not table in col_table_map[col.sql().upper()]:
                similar = find_similar_col(col, val, col_table_map, eq_pairs)

                if similar and (invalid_ident == None or col.sql().upper() == invalid_ident.upper()):
                    similar_col = similar[0]
                    similar_val = similar[1]                    
                    col.replace(similar_col)                    
                    val.replace(similar_val)

                    for select_col in select_cols:
                        if col.sql().upper() == select_col.sql().upper():
                            select_col.replace(similar_col)

                    updated = True

    if updated:
        return ast.sql(dialect=db_type)
    else:
        return None

def find_aliases(ast):
    alias_nodes = ast.find_all(sqlglot.expressions.Alias)
    aliases = set()

    for alias in alias_nodes:
        aliases.add(alias.alias)

    return aliases

def find_similar_col(col, val, col_table_map, eq_pairs):
    if len(val.sql()) < 3:
        return None

    for other_col, other_val in eq_pairs:
        if col.sql().upper() == other_col.sql().upper() and \
           val.sql().upper() == other_val.sql().upper():
            continue

        if other_col.sql().upper() in col_table_map and val.this.upper() in other_val.this.upper():
            return (other_col, other_val)

    return None

def find_eqs(ast):
    eqs = ast.find_all(sqlglot.expressions.EQ)
    eq_pairs = set()

    for eq in eqs:
        lhs = eq.this
        rhs = eq.expression

        if isinstance(lhs, sqlglot.expressions.Column) and \
           isinstance(rhs, sqlglot.expressions.Literal):
            eq_pairs.add((lhs, rhs))

    return eq_pairs

def fix_invalid_col(sql, col_table_map, db_type, error_msg=''):
    invalid_ident = None

    if db_type == 'snowflake' and error_msg != '':
        if 'invalid identifier' in error_msg:
            pattern = r"invalid identifier '([^']+)'"
            match = re.search(pattern, error_msg)

            if match:
                invalid_ident = match.group(1)
                if '.' in invalid_ident:
                    return sql
        else:
            return sql
    
    try:
        ast_list = sqlglot.parse(sql, dialect=db_type)
    except:
        return sql

    if len(ast_list) > 1:
        return sql

    ast = ast_list[0]

    if ast == None: # this is when sql is just ;
        return sql
    
    fixed = fix_invalid_col_in_select(ast, col_table_map, invalid_ident, db_type)

    if fixed:
        return fixed
    else:
        return sql

# This is a snowflake specific fix for now
# This autofix is currently only invoked with gpt4
# Sometimes a query has multiple statements like
#    SELECT a FROM b; SELECT c FROM d; ...
# The first one is always not needed/incorrect, so remove the first one
# If end result has multiple SELECTs, then must wrap them in BEGIN ... END
def fix_stmt_count(sql, db_type):
    if db_type != 'snowflake':
        return sql

    try:
        ast_list = sqlglot.parse(sql, dialect=db_type)
    except:
        return sql

    if len(ast_list) <= 1 or None in ast_list:
        return sql

    if len(ast_list) == 2:
        new_sql = ast_list[-1].sql(dialect=db_type)
    else:
        new_sql_list = []
        for i in range(1, len(ast_list)):
            new_sql_list.append(ast_list[i].sql(dialect=db_type))
        new_sql = 'BEGIN ' + '; '.join(new_sql_list) + '; END'

    return new_sql
