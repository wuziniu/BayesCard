import ast
import numpy as np

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}


def str_pattern_matching(s):
    # split the string "attr==value" to ('attr', '=', 'value')
    op_start = 0
    if len(s.split(' IN ')) != 1:
        s = s.split(' IN ')
        attr = s[0].strip()
        try:
            value = list(ast.literal_eval(s[1].strip()))
        except:
            temp_value = s[1].strip()[1:][:-1].split(',')
            value = []
            for v in temp_value:
                value.append(v.strip())
        return attr, 'in', value

    for i in range(len(s)):
        if s[i] in OPS:
            op_start = i
            if i + 1 < len(s) and s[i + 1] in OPS:
                op_end = i + 1
            else:
                op_end = i
            break
    attr = s[:op_start]
    value = s[(op_end + 1):].strip()
    ops = s[op_start:(op_end + 1)]
    try:
        value = list(ast.literal_eval(s[1].strip()))
    except:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                value = value
    return attr.strip(), ops.strip(), value


def parse_query_single_table(query, column_names, epsilon=1.0):
    useful = query.split(' WHERE ')[-1].strip()
    n_attrs = len(column_names)
    query_l = np.zeros(n_attrs) - np.infty
    query_r = np.zeros(n_attrs) + np.infty
    for sub_query in useful.split(' and '):
        attr, ops, value = str_pattern_matching(sub_query.strip())
        i = column_names.index(attr)
        if ops == "==" or ops == "=":
            query_l[i] = value
            query_r[i] = value
        elif ops == ">=":
            query_l[i] = value
        elif ops == ">":
            query_l[i] = value+epsilon
        elif ops == "<=":
            query_r[i] = value
        elif ops == "<":
            query_r[i] = value - epsilon
    return query_l, query_r

