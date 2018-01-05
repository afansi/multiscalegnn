import copy


def tablelength(T):
    return len(T) if T is not None else 0


def shallowcopy(orig):
    return copy.copy(orig)


def tablemerge(t1, t2):
    t1.extend(t2)
    return t1


def settodict(s):
    return {a: True for a in s}


def pairs(test):
    iterator = None
    if isinstance(test, list):
        iterator = enumerate(test)
    elif isinstance(test, dict):
        iterator = test.items()
    elif isinstance(test, set):
        iterator = settodict(test).items()

    return iter(iterator)


def tablefuse(test):
    hash_v = set()
    res = []
    for k, v in pairs(test):
        if v not in hash_v:
            res.append(v)  # you could print here instead of saving to resul
            hash_v.add(v)
    return res


def setNew(t):
    res = set()
    for k, v in pairs(t):
        res.add(v)

    return res


def settotable(a):
    return list(a)


def setContains(s, key):
    if key not in s:
        return False

    if isinstance(s, dict):
        return s[key] is not None

    return True


def tableunion_size(ta, tb):
    a = setNew(ta)
    b = setNew(tb)
    c = a.union(b)
    return tablelength(c), tablelength(a), tablelength(b)


def tableintersection_size(ta, tb):
    a = setNew(ta)
    b = setNew(tb)
    res = a.intersection(b)
    return tablelength(res)


def tableunion(ta, tb):
    a = setNew(ta)
    b = setNew(tb)
    return a.union(b)


def tableintersection(ta, tb):
    a = setNew(ta)
    b = setNew(tb)
    return a.intersection(b)


def setintersection(a, b):
    return a.intersection(b)


def tablemin(t):
    if len(t) == 0:
        return None, None
    key, value = 0, t[0]
    for i in range(1, len(t)):
        if value > t[i]:
            key, value = i, t[i]
    return key, value


def spairs(t, key=None):
    if isinstance(t, list):
        return iter(enumerate(t))

    keys = list(t.keys())

    keys = sorted(keys, key=key)

    for k in keys:
        yield k, t[k]


def tablesampleg(t1, t2, Csize):
    set2 = setNew(t2)
    tmp = {}
    for _, l in pairs(t1):
        if not setContains(set2, l):
            tmp[l] = Csize[l]

    stmp = spairs(tmp, key=lambda a: tmp[a])

    return stmp


def tablesample(t1, t2, Csize, targetsize):
    set2 = setNew(t2)
    tmp = {}
    pos = None
    for _, l in pairs(t1):
        if not setContains(set2, l):
            tmp[l] = abs(Csize[l] - targetsize)
            refval = tmp[l]
            pos = l

    for i, l in pairs(tmp):
        if l < refval:
            refval = l
            pos = i

    return pos


def inverttablekeys(test):

    r = {}
    count = 0
    for k, v in pairs(test):
        r[k] = count
        count = count + 1

    return r


# Function to convert a table to a string
# Metatables not followed
# Unless key is a number it will be taken and converted to a string
def t2s(t):
    # local levels = 0
    # Table to track recursion into nested tables (cL = current recursion
    # level)
    rL = {'cL': 1}
    rL[rL['cL']] = {}
    result = []

    rL[rL['cL']]['_f'] = pairs(t)
    # result[len(result) + 1] =  "{\n \t"+str(levels+1)
    result[len(result)] = "{"        # Non pretty version
    rL[rL['cL']]['t'] = t
    while True:
        k, v = next(rL[rL['cL']]['_f'], (None, None))
        if k is None and rL['cL'] == 1:
            break
        elif k is None:
            # go up in recursion level
            # If condition for pretty printing
            # if result[  # result]:sub(-1,-1) == "," then
            #      result[  # result] = result[#result]:sub(1,-3)    -- remove the tab and the comma
            # else
            #      result[  # result] = result[#result]:sub(1,-2)    -- just remove the tab
            # end
            result.append("},")    # non pretty version
            # levels = levels - 1
            rL['cL'] = rL['cL'] - 1
            rL[rL['cL'] + 1] = None
        else:
            # Handle the key and value here
            if isinstance(k, int):
                result.append("[" + str(k) + "]=")
            else:
                result.append("[\"" + str(k) + "\"]=")

            if type(v) in [list, dict, set]:
                # Check if this is not a recursive table
                goDown = True
                for i in range(1, rL['cL']):
                    if id(v) == id(rL[i]['t']):
                        # This is recursive do not go down
                        goDown = False
                        break

                if goDown:
                    # Go deeper in recursion
                    # levels = levels + 1
                    rL['cL'] = rL['cL'] + 1
                    rL[rL['cL']] = {}
                    rL[rL['cL']]['_f'] = pairs(v)
                    # result[  # result + 1] =
                    # "{\n"..string.rep("\t",levels+1)
                    result.append("{")    # non pretty version
                    rL[rL['cL']]['t'] = v
                else:
                    # result[  # result + 1] =
                    # "\""..tostring(v).."\",\n"..string.rep("\t",levels+1)
                    result.append(
                        "\"" + str(v) + "\",")    # non pretty version

            elif type(v) in [int, bool]:
                # result[  # result + 1] =
                # tostring(v)..",\n"..string.rep("\t",levels+1)
                result.append(str(v) + ",")    # non pretty version
            else:
                # result[  # result + 1] =
                # string.format("%q",tostring(v))..",\n"..string.rep("\t",levels+1)
                result.append(
                    "\"" + str(v) + "\",")    # non pretty version

    result.append("}")    # non pretty version
    return ''.join(result)


# Function to convert a table to a string with indentation for pretty printing
# Metatables not followed
# Unless key is a number it will be taken and converted to a string
def t2spp(t):
    levels = 0
    # Table to track recursion into nested tables (cL = current recursion
    # level)
    rL = {'cL': 1}
    rL[rL['cL']] = {}
    result = []

    rL[rL['cL']]['_f'] = pairs(t)
    result.append("{\n\t" + str(levels + 1))
    rL[rL['cL']]['t'] = t
    while True:
        k, v = next(rL[rL['cL']]['_f'], (None, None))
        if k is None and rL['cL'] == 1:
            break
        elif k is None:
            # go up in recursion level
            # If condition for pretty printing
            if result[-1][-1] == ",":
                result[-1] = result[-1][0:-3]  # remove the tab and the comma
            else:
                result[-1] = result[-1][0:-2]  # just remove the tab

            levels = levels - 1
            rL['cL'] = rL['cL'] - 1
            rL[rL['cL'] + 1] = None
            # for pretty printing
            result.append("},\n\t" + str(levels + 1))
        else:
            # Handle the key and value here
            if isinstance(k, int):
                result.append("[" + str(k) + "]=")
            else:
                result.append("[\"" + str(k) + "\"]=")

            if type(v) in [list, dict, set]:
                # Check if this is not a recursive table
                goDown = True
                for i in range(1, rL['cL']):
                    if id(v) == id(rL[i]['t']):
                        # This is recursive do not go down
                        goDown = False
                        break

                if goDown:
                    # Go deeper in recursion
                    levels = levels + 1
                    rL['cL'] = rL['cL'] + 1
                    rL[rL['cL']] = {}
                    rL[rL['cL']]['_f'] = pairs(v)

                    # For pretty printing
                    result.append("{\n\t" + str(levels + 1))
                    rL[rL['cL']]['t'] = v
                else:
                    result.append("\"" + str(v) + "\",\n\t" +
                                  str(levels + 1))    # For pretty printing

            elif type(v) in [int, bool]:
                # For pretty printing
                result.append(str(v) + ",\n\t" + str(levels + 1))
            else:
                result.append("\"" + str(v) + "\"" + ",\n\t" +
                              str(levels + 1))        # For pretty printing
    # If condition for pretty printing
    if result[-1][-1] == ",":
        result[-1] = result[-1][0:-3]  # remove the tab and the comma
    else:
        result[-1] = result[-1][0:-2]    # just remove the tab

    result.append("}")
    return ''.join(result)


# Function to convert a table to string following the recursive tables also
# Metatables are not followed
def t2sr(t):
    if not (type(t) in [list, dict, set]):
        return None, 'Expected table parameter'

    # Table to track recursion into nested tables (cL = current recursion
    # level)
    rL = {'cL': 1}
    rL[rL['cL']] = {}
    # Table to store a list of tables indexed into a string and their variable
    # name
    tabIndex = {}
    latestTab = 0
    result = []

    rL[rL['cL']]['_f'] = pairs(t)
    result.append('t0={}')    # t0 would be the main table
    rL[rL['cL']]['t'] = t
    rL[rL['cL']]['tabIndex'] = 0
    tabIndex[id(t)] = rL[rL['cL']]['tabIndex']
    while True:
        key = None
        k, v = next(rL[rL['cL']]['_f'], (None, None))

        if k is None and rL['cL'] == 1:
            break
        elif k is None:
            # go up in recursion level
            rL['cL'] = rL['cL'] - 1
            if ('vNotDone' in rL[rL['cL']]) and (
                    rL[rL['cL']]['vNotDone'] is not None):
                key = 't' + str(rL[rL['cL']]['tabIndex']) + \
                    '[t' + str(rL[rL['cL'] + 1]['tabIndex']) + ']'

                result.append("\n" + key + "=")
                v = rL[rL['cL']]['vNotDone']

            rL[rL['cL'] + 1] = None
        else:
            # Handle the key and value here
            if isinstance(k, int):
                key = 't' + str(rL[rL['cL']]['tabIndex']) + '[' + str(k) + ']'
                result.append("\n" + key + "=")
            elif isinstance(k, str):
                key = 't' + str(rL[rL['cL']]['tabIndex']) + '.' + str(k)
                result.append("\n" + key + "=")
            else:
                # Table key
                # Check if the table already exists
                if id(k) in tabIndex:
                    key = 't' + str(rL[rL['cL']]['tabIndex']) + \
                        '[t' + str(tabIndex[id(k)]) + ']'
                    result.append("\n" + key + "=")
                else:
                    # Go deeper to stringify this table
                    latestTab = latestTab + 1
                    # rL[rL.cL].str =
                    # rL[rL.cL].str..'\\nt'..tostring(latestTab)..'={}'
                    result.append("\nt" + str(latestTab) + "={}")
                    rL[rL['cL']]['vNotDone'] = v
                    rL['cL'] = rL['cL'] + 1
                    rL[rL['cL']] = {}
                    rL[rL['cL']]['_f'] = pairs(k)
                    rL[rL['cL']]['tabIndex'] = latestTab
                    rL[rL['cL']]['t'] = k
                    tabIndex[id(k)] = rL[rL['cL']]['tabIndex']

        if key is not None:
            rL[rL['cL']]['vNotDone'] = None
            if type(v) in [list, dict, set]:
                # Check if this table is already indexed
                if id(v) in tabIndex:
                    # rL[rL.cL].str = rL[rL.cL].str..'t'..tabIndex[v]
                    result.append('t' + str(tabIndex[id(v)]))
                else:
                    # Go deeper in recursion
                    latestTab = latestTab + 1

                    result.append(
                        "{}\nt" +
                        str(latestTab) +
                        '=' +
                        key)        # New table
                    rL['cL'] = rL['cL'] + 1
                    rL[rL['cL']] = {}
                    rL[rL['cL']]['_f'] = pairs(v)
                    rL[rL['cL']]['tabIndex'] = latestTab
                    rL[rL['cL']]['t'] = v
                    tabIndex[id(v)] = rL[rL['cL']]['tabIndex']

            elif isinstance(v, int):
                result.append(str(v))
            elif isinstance(v, bool):
                result.append(str(v))
            else:
                result.append('"' + str(v) + '"')
    return ''.join(result)


def compareTables(t1, t2):
    for k, v in pairs(t1):
        # print(k,v)
        if type(v) in [int, str, float, bool] or callable(v):
            if isinstance(t2, list):
                if isinstance(k, int):
                    if len(t2) >= k:
                        return False
                    else:
                        if not(v == t2[k]):
                            return False
                else:
                    return False
            else:
                if k not in t2:
                    return False
                if isinstance(t2, dict) and not (v == t2[k])	:
                    return False
        else:
            if isinstance(t2, list):
                if isinstance(k, int):
                    if len(t2) >= k:
                        return False
                    else:
                        if not compareTables(v, t2[k]):
                            return False
                else:
                    return False

            else:
                if k not in t2:
                    return False
                if isinstance(t2, dict) and not compareTables(v, t2[k]):
                    return False
    return True
