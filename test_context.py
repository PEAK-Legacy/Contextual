import sys, unittest
suites = []

if sys.version>='2.4':
    suites.append('README.txt')   
    if sys.version>='2.5':
        suites.append('Contextual.txt')    

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



















class TestStateInitialization(unittest.TestCase):
    
    def testStateGet(self):
        from peak.context import State
        self.assertEqual(State.parent, State.root)

    def testStateLookupInOtherThread(self):
        from peak.context import State, Service, lookup
        my_state = State.get()
        other_state = []
        def test_other_thread():
            lookup(Service) # test lookup() path of state creation
            other_state.append(State.get())
        from threading import Thread
        t = Thread(target = test_other_thread)
        t.start()
        t.join()
        state = other_state.pop()
        self.assertNotEqual(my_state, state)
        self.assertEqual(state.parent, State.root)


def additional_tests():
    import doctest
    import __future__
    globs = dict(sorted=sorted)
    if hasattr(__future__,'with_statement'):
        globs['with_statement'] = __future__.with_statement
    return doctest.DocFileSuite(
        optionflags=doctest.ELLIPSIS|doctest.REPORT_ONLY_FIRST_FAILURE,
        globs=globs, *suites
    )









