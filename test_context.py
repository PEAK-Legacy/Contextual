import sys
suites = []

if sys.version>='2.4':
    suites.append('README.txt')   
    if sys.version>='2.5':
        suites.append('Contextual.txt')    
    #XXX suites.append('context.txt')

suites.append('context_tests.txt')

try:
    sorted = sorted
except NameError:
    def sorted(seq,key=None):
        if key:
            d = [(key(v),v) for v in seq]
        else:
            d = list(seq)
        d.sort()
        if key:
            return [v[1] for v in d]
        return d

def additional_tests():
    import doctest
    import __future__
    globs = dict(sorted=sorted)
    if hasattr(__future__,'with_statement'):
        globs['with_statement'] = __future__.with_statement
    return doctest.DocFileSuite(
        #'README.txt', 'Contextual.txt', 'context.txt', 'context_tests.txt',
        optionflags=doctest.ELLIPSIS|doctest.REPORT_ONLY_FIRST_FAILURE,
        globs=globs, *suites
    )

