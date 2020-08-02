import itertools
import copy
import ast
def cartesian_product(d):
    target_list = []
    for key in d:
        val = d[key]
        if type(val) != list:
            val = [val]
        target_list.append(val)
    return itertools.product(*target_list)

def str_pattern_matching(condition):
    # split the string "attr==value" to ('attr', '=', 'value')
    s = copy.deepcopy(condition)
    op = ['>', '<', '=']
    op_start = 0
    if len(s.split(' IN ')) != 1:
        s = s.split(' IN ')
        attr = s[0].strip()
        try:
            value = list(ast.literal_eval(s[1].strip()))
        except:
            value = s[1].strip()[1:][:-1].split(',')
        return attr, value

    for i in range(len(s)):
        if s[i] in op:
            op_start = i
            if i + 1 < len(s) and s[i + 1] in op:
                op_end = i + 1
            else:
                op_end = i
            break
    value = s[(op_end+1):]
    try:
        value = float(value)
    except:
        try:
            value = list(ast.literal_eval(s[1].strip()))
        except:
            value = value

    return s[:op_start], value

def convert_to_pandas_query(query):
    query_str = ""
    n_cols = 0
    for attr in query:
        query_str += attr
        if len(query[attr]) == 1:
            query_str += (' == ' + str(query[attr][0]))
        else:
            query_str += (' in ' + str(query[attr]))
        if n_cols != 0:
            query_str = ' and ' + query_str
        n_cols += 1
    return query_str
