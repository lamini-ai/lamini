import json
import re
import sqlglot

def extract_sql_part(query):
    i_codeblock = query.find('```')

    if i_codeblock != -1:
        query = query[i_codeblock:]

    queryl = query.lower()

    if not query.lower().startswith('with'):
        init_tokens = ['select', 'with']
        i_start = None

        for init_token in init_tokens:
            i_start = queryl.find(init_token)
            if i_start != -1:
                break
            if not i_start:
                return None

        query = query[i_start:]

    i_semicolon = query.find(';')

    if i_semicolon != -1:
        query = query[:i_semicolon + 1]    

    return query

def fix_invalid_col_in_select(ast, col_table_map, invalid_ident):
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
        return ast.sql()
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

# For some reason, the sqlglot library tends to automatically add `NULLS LAST` at
# the end of the query.  This doesn't change syntax or semantic of the SQL.
# We remove `NULL LAST` here, even though it doesn't break anything
def strip_special_suffix(query, original_sql):
    if original_sql.endswith('NULLS LAST') or original_sql.endswith('NULLS LAST;') or \
       original_sql.endswith('nulls last') or original_sql.endswith('nulls last;'):
        return query
    if query.endswith('NULLS LAST'):
        query = query.rstrip('NULLS LAST')
    elif query.endswith('nulls last'):
        query = query.rstrip('nulls last')
    elif query.endswith('NULLS LAST;'):
        query = query.rstrip('NULLS LAST;') + ';'
    elif query.endswith('nulls last;'):
        query = query.rstrip('nulls last;') + ';'        

    return query

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
    
    fixed = fix_invalid_col_in_select(ast, col_table_map, invalid_ident)

    if fixed:
        fixed = strip_special_suffix(fixed, sql)
        return fixed
    else:
        return sql
