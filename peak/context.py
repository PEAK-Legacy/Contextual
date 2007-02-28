import sys, new
from thread import get_ident
from peak.util.symbols import NOT_GIVEN
from peak.util.proxies import ObjectWrapper
from peak.util.decorators import rewrap, cache_source

__all__ = [
    'Service', 'replaces', 'setting', 'RuleConflict', 'DynamicRuleError'
    'State', 'Action', 'resource', 'namespace', 'parameter', 'using', 'only',
    'lookup', 'manager', 'reraise', 'with_', 'call_with',
]


def _clonef(src, impl, name=None):
    """Create a copy of function `impl` that looks like `src`"""
    f = new.function(
        impl.func_code, src.func_globals, name or src.__name__,
        impl.func_defaults, impl.func_closure
    )
    f.__doc__  = src.__doc__
    f.__dict__ = src.__dict__.copy()
    return f

def qname(f):
    if hasattr(f,'__module__'):
        m = f.__module__
    else:
        m = f.func_globals.get('__name__')
    if m:
        return '%s.%s' % (m,f.__name__)
    return f.__name__


class RuleConflict(Exception):
    """Attempt to set a rule that causes a visible conflict in the state"""

class DynamicRuleError(Exception):
    """A fallback or wildcard rule attempted to access dynamic state"""



class State(object):
    """A thread's current configuration and state"""

    def __new__(cls, *rules, **attrs):
        return attrs and object.__new__(cls) or empty()

    def __init__(self, *rules, **attrs):
        """Create an empty state with `rules` in effect"""
        self.__dict__.update(attrs)
        # for rule in rules: self += rule

    def __getitem__(self, key):
        """Get the rule for `key`"""
        return self.getRule(key)

    def __setitem__(self, key, rule):
        """Set the rule for `key`"""
        return self.setRule(key, rule)

    def swap(self):
        """Make this state current and return the old one"""
        # this method is replaced on each instance!
        raise NotImplementedError

    def child(self, *rules):
        """Return a new child state of this one, with `rules` in effect"""
        # this method is replaced on each instance!
        raise NotImplementedError

    def get(key=None):
        """Return the current state (no args) or a current rule (w/key)"""
        # this method is replaced later below
        raise NotImplementedError

    get = staticmethod(get)






class using(object):
    """Thing that can be added to a state to set its rules"""

    __slots__ = ['_saved_state', '__rules__']

    def __init__(self, *rules):
        self.__rules__ = ()
        for r in rules:
            self.__rules__ += r.__rules__

    def apply_to(self, state):
        for key, val in self.__rules__:
            state[key] = val
        return state.swap()

    def __enter__(self):
        self._saved_state = self.apply_to(State.get().child())

    def __exit__(self, typ, val, tb):
        self._saved_state.swap()
        del self._saved_state


class only(using):
    def __enter__(self):
        self._saved_state = self.apply_to(State())

class _using_var(using):
    """Sets a single rule"""
    def __init__(self, key, value):
        self.__rules__ = (key, value),


'''with service_inst:
with variable(value):
with context.using( service_inst, variable(val), other_var(other_val) ):
with context.only(
    service_inst, variable(val), other_var(other_val)
):
'''

noop = lambda rule: rule
mngr = lambda rule: Action.manage(rule())

def setting(func):
    """Decorator to create a configuration setting from a function"""
    return make_var(func, State.get, noop)

def parameter(func):
    """Decorator that makes a function into a dynamic parameter"""
    return make_var(_clonef(func, lambda: func), lookup, apply)

def resource(func):
    """Decorator to create a configuration setting from a function
    """
    return make_var(_clonef(func, lambda: func), lookup, mngr)


def make_var(func, lookup, application, name=None):

    NG = NOT_GIVEN
    UV = _using_var

    def impl(value=NG):
        if value is NG:
            return lookup(impl)
        else:
            return UV(impl, value)

    def fallback(inherit, key):
        if inherit is None:
            return func()
        else:
            return inherit(key)

    impl = _clonef(func, impl, name)
    impl.__fallback__ = fallback
    impl.__apply__ = application
    impl.__lookup__ = lookup
    return impl


def _let_there_be_state():
    """Create a world of states, manipulable only by exported functions"""

    states = {}

    def _swap(what):
        this_thread = get_ident()
        old = states.setdefault(this_thread, what)
        states[this_thread] = what
        return old

    def lookup(key):
        """Return the value of `key` in the current state"""
        try:
            state, getRule, lookup = states[get_ident()]
        except KeyError:
            new_state().swap()
            state, getRule, lookup = states[get_ident()]
        return lookup(key)

    def get(key=None):
        try:
            state, getRule, lookup = states[get_ident()]
        except KeyError:
            new_state().swap()
            state, getRule, lookup = states[get_ident()]
        if key is None:
            return state
        return getRule(key)

    def disallow(key):
        raise DynamicRuleError(
            "default rule tried to read dynamic state", key
        )

    def empty():
        """Return a new, empty State instance"""
        return new_state()



    def new_state(inherit=None, inheritedDistances=None, propagate=None):
        buffer = {}
        rules = {}
        values = {}
        distances = {}
        computing = {}
        get_stack, set_stack = computing.get, computing.setdefault

        def getRule(key):
            """Get my rule for `key`"""
            try:
                rule = rules[key]
            except KeyError:
                try:
                    rule = buffer[key]
                except KeyError:
                    rule = buffer.setdefault(key, __fallback(key))

                # snapshot the currently-set value - this is thread-safe
                # because setdefault is atomic, so rules[key] will only *ever*
                # have one value, even if it's not the one now in the buffer
                #
                rule = rules.setdefault(key, rule)
                if key not in distances:
                    if inherit and inherit(key)==rule:
                        distances.setdefault(key, inheritedDistances[key]+1)
                    else:
                        distances[key] = 0

            if computing:
                # Ensure that any value being computed in this thread has a
                # maximum propagation distance no greater than this rule's
                # distance.
                stack = get_stack(get_ident())
                if stack:
                    stack[-1] = min(stack[-1], distances[key])

            return rule



        def setRule(key, rule):
            """Set my rule for `key`, or raise an error if inconsistent"""
            buffer[key] = rule
            # as long as a snapshot hasn't been taken yet, `old` will be `rule`
            old = rules.get(key, rule)
            if old is not rule and old != rule:
                raise RuleConflict(key, old, rule)

        def getValue(key):
            """Get the dynamic value of `key`"""
            try:
                value = values[key]

            except KeyError:
                this_thread = get_ident()
                rule = getRule(key)     # this ensures distances[key] is known

                stack = set_stack(this_thread, [])
                stack.append(distances[key])

                try:
                    value = key.__apply__(rule)
                finally:
                    distance = stack.pop()
                    if not stack:
                        del computing[this_thread]
                    else:
                        stack[-1] = min(stack[-1], distance)

                value = publish(distance, key, value)

            else:
                if computing:
                    stack = get_stack(get_ident())
                    if stack:
                        stack[-1] = min(stack[-1], distances[key])

            return value



        def publish(distance, key, value):
            """Accept value from this state or child, and maybe propagate it"""

            # It's safe to update distances here because no thread can depend
            # on the changed distance for propagation unless *some* thread has
            # already finished the relevant computation -- i.e., done this very
            # update.  Otherwise, there would be no values[key], therefore the
            # thread would have to follow the same code path, ending up by
            # being the thread doing this update!  Ergo, this is a safe update.
            #
            distances[key] = distance

            if distance and propagate:
                # pass it up to the parent, but use the value the parent has.
                # therefore, the parent at "distance" height from us will be
                # the arbiter of the value for all its children, ensuring that
                # exactly one value is used for the entire subtree!
                #
                value = propagate(distance-1, key, value)

            # Return whatever value wins at this level to our children
            return values.setdefault(key, value)


        def child():
            """Return a new child state"""
            s = new_state(getRule, distances, publish)
            s.parent = this
            return s

        def __fallback(key):
            """Compute the fallback for key"""
            old = _swap(disabled)
            try:
                return key.__fallback__(inherit, key)
            finally:
                _swap(old)




        def swap():
            state, get, lookup = _swap(enabled)
            if lookup is disallow:
                _swap((state, get, lookup))
                raise DynamicRuleError(
                    "default rule tried to change states"
                )
            return state

        this = State(getRule=getRule, setRule=setRule, swap=swap, child=child)

        enabled  = this, getRule, getValue
        disabled = this, getRule, disallow

        return this

    return lookup, staticmethod(get), empty


lookup, State.get, empty = _let_there_be_state()
del _let_there_be_state




















class _GeneratorContextManager(object):
    """Helper for @context.manager decorator."""

    def __init__(self, gen):
        self.gen = gen

    def __enter__(self):
        for value in self.gen:
            return value
        else:
            raise RuntimeError("generator didn't yield")

    def __exit__(self, typ, value, traceback):
        if typ is None:
            for value in self.gen:
                raise RuntimeError("generator didn't stop")
        else:
            try:
                old = _swap_exc_info((typ, value,traceback))
                try:
                    self.gen.next()
                finally:
                    _swap_exc_info(old)

                raise RuntimeError("generator didn't stop after throw()")

            except StopIteration, exc:
                # Suppress the exception *unless* it's the same exception that
                # was passed to throw().  This prevents a StopIteration
                # raised inside the "with" statement from being suppressed
                return exc is not value
            except:
                # only re-raise if it's *not* the exception that was
                # passed to throw(), because __exit__() must not raise
                # an exception unless __exit__() itself failed.  But throw()
                # has to raise the exception to signal propagation, so this
                # fixes the impedance mismatch between the throw() protocol
                # and the __exit__() protocol.
                if sys.exc_info()[1] is not value:
                    raise

def manager(func):
    """Emulate 2.5 ``@contextlib.contextmanager`` decorator"""
    gcm = _GeneratorContextManager
    return rewrap(func, lambda *args, **kwds: gcm(func(*args, **kwds)))

def with_(ctx, func):
    """Perform PEP 343 "with" logic for Python versions <2.5

    The following examples do the same thing at runtime::

        Python 2.5+          Python 2.3/2.4
        ------------         --------------
        with x as y:         z = with_(x,f)
            z = f(y)

    This function is used to implement the ``call_with()`` decorator, but
    can also be used directly.  It's faster and more compact in the case where
    the function ``f`` already exists.
    """
    inp = ctx.__enter__()
    try:
        retval = func(inp)
    except:
        if not ctx.__exit__(*sys.exc_info()):
            raise
    else:
        ctx.__exit__(None, None, None)
        return retval

def reraise():
    """Reraise the current contextmanager exception, if any"""
    typ,val,tb = _exc_info.get(get_ident(), nones)
    if typ:
        try:
            raise typ,val,tb
        finally:
            del typ,val,tb




def call_with(ctxmgr):
    """Emulate the PEP 343 "with" statement for Python versions <2.5

    The following examples do the same thing at runtime::

        Python 2.5+          Python 2.4
        ------------         -------------
        with x as y:         @call_with(x)
            print y          def do_it(y):
                                 print y

    ``call_with(foo)`` returns a decorator that immediately invokes the
    function it decorates, passing in the same value that would be bound by
    the ``as`` clause of the ``with`` statement.  Thus, by decorating a
    nested function, you can get most of the benefits of "with", at a cost of
    being slightly slower and perhaps a bit more obscure than the 2.5 syntax.

    Note: because of the way decorators work, the return value (if any) of the
    ``do_it()`` function above will be bound to the name ``do_it``.  So, this
    example prints "42"::

        @call_with(x)
        def do_it(y):
            return 42

        print do_it

    This is rather ugly, so you may prefer to do it this way instead, which
    more explicitly calls the function and gets back a value::

        def do_it(y):
            return 42

        print with_(x, do_it)
    """
    return with_.__get__(ctxmgr, type(ctxmgr))





_exc_info = {}
nones = None, None, None

def _swap_exc_info(data):
    this_thread = get_ident()
    old = _exc_info.get(this_thread, nones)
    _exc_info[this_thread] = data
    return old


def redirect_attribute(cls, name, payload):
    meta = type(cls)
    if getattr(meta, '__for_class__', None) is not cls:
        cls.__class__ = meta = type(meta)(
            cls.__name__+'Class', (meta,), {'__module__':cls.__module__, '__for_class__':cls}
        )
        # XXX activate_attrs(meta)?

    setattr(meta, name, property(
        lambda s: getattr(s.get(), name),
        lambda s,v: setattr(s.get(), name, v),
        lambda s: delattr(s.get(), name),
    ))

_ignore = {
    '__name__':1, '__module__':1, '__return__':1, '__slots__':1, 'get':1,
    '__init__':1, '__metaclass__':1, '__doc__':1, '__call__': 1, '__new__':1,
}.get













class ServiceClass(type):

    def __new__(meta, name, bases, cdict):
        cls = super(ServiceClass, meta).__new__(meta, name, bases, cdict)
        if 'get' not in cdict:
            cls.get = staticmethod(classmethod(lookup).__get__(None, cls))
        for k, v in cdict.items():
            if not isinstance(v, (classmethod,staticmethod))and not _ignore(k):
                redirect_attribute(cls, k, v)
        return cls


class Service(using):
    """A replaceable, thread-local singleton"""

    __slots__ = ()  # pure mixin class
    __metaclass__ = ServiceClass

    __rules__ = property(lambda self: ((self.get.im_self, self.getself),))

    def getself(self): return self

    def __init__(self):
        pass

    def __default__(cls):
        return cls()

    __default__ = classmethod(__default__)

    def __fallback__(cls, inherit, key):
        if inherit: return inherit(key)
        return cls.__default__

    __fallback__ = classmethod(__fallback__)

    __apply__ = staticmethod(apply)




class Action(Service):
    """Service for managing transaction-scoped resources"""

    def __init__(self):
        self.managers = None
        self.status = {}

    def __default__(cls):
        raise RuntimeError("No Action is currently active")
    __default__ = classmethod(__default__)

    def __enter__(self):
        if self.managers is not None:
            raise RuntimeError("Action is already in use")
        self.managers = []
        return Service.__enter__(self)

    def __exit__(self, *exc):
        if self.managers is None:
            raise RuntimeError("Action is not currently in use")

        Service.__exit__(self, *exc)
        managers, self.managers = self.managers, None
        while managers:
            managers.pop().__exit__(*exc)  # XXX how do we handle errors?

    def manage(self, ob):
        try:
            enter = ob.__enter__
        except AttributeError:
            return ob
        ctx = ob
        ob = ctx.__enter__()
        # don't call __exit__ unless __enter__ succeeded
        # (if there was an error, we wouldn't have gotten this far)
        self.managers.append(ctx)
        return ob

        # XXX return self.manage(key, ob)


class namespace(ObjectWrapper):
    """Decorator that wraps a setting or resource w/an extensible namespace"""

    __slots__ = '__dict__'

    def __getattr__(self, key):
        try:
            return nsd(self)[key]
        except KeyError:
            if key.startswith('__') and key.endswith('__'):
                return getattr(self.__subject__, key)
            return self[key]

    def __getitem__(self,key):
        if '.' in key:
            for key in key.split('.'):
                self = self[key]
            return self
        try:
            return nsd(self)[key]
        except KeyError:
            # TODO: verify syntax of key: nonempty, valid chars, ...?
            me = self.__subject__
            impl = make_var(me, me.__lookup__, me.__apply__, "%s.%s" % (self.__name__, key))

            if key=='*':
                if hasattr(me, '__namespace__'):
                    ns = me.__namespace__
                    impl.__fallback__ = lambda i,k: State.get(ns['*'])
                else:
                    impl.__fallback__ = lambda i,k: default_fallback(me)
            else:
                impl.__fallback__ = lambda i,k: State.get(self['*'])(i,k)

            impl.__namespace__ = self
            nsd(self)[key] = impl = type(self)(impl)
            return impl




    def __contains__(self,key):
        d = nsd(self)
        if key in d:
            return True
        if '.' in key:
            for key in key.split('.'):
                if key not in d:
                    return False
                d = nsd(d[key])
            else:
                return True
        return False

    def __iter__(self):
        for key in nsd(self):
            if key!='*':
                yield key

nsd = namespace.__dict__['__dict__'].__get__

def default_fallback(me):
    """Fallback used for a namespace's top-level settings' wildcard results"""

    def fallback(inherit, key):
        if inherit is None:
            return me()
        return inherit(key)

    return fallback












class Source(object):
    """Object representing a source file (or pseudo-file)"""

    __slots__ = "filename", "__weakref__"

    def __init__(self, filename, source=None):
        global linecache; import linecache
        self.filename = filename
        if source is not None:
            cache_source(filename, source, self)

    def compile(self, *args, **kw):
        return Line(''.join(self), self, 1).compile(*args, **kw)

    def __getitem__(self, key):
        return Line(linecache.getlines(self.filename)[key], self, key+1)

    def __repr__(self):
        return "Source(%r)" % self.filename

    def recode(self, code, offset=0):
        if not isinstance(code, new.code):
            return code
        return new.code(
            code.co_argcount, code.co_nlocals, code.co_stacksize,
            code.co_flags, code.co_code,
            tuple([self.recode(c, offset) for c in code.co_consts]+[self]),
            code.co_names, code.co_varnames, code.co_filename, code.co_name,
            code.co_firstlineno+offset, code.co_lnotab, code.co_freevars,
            code.co_cellvars
        )










class Line(str):
    """String that knows its file and line number"""

    def __new__(cls, text, source, line):
        return str.__new__(cls, text)

    def __init__(self, text, source, line):
        self.source = source
        self.line = line

    def compile(self, *args, **kw):
        # XXX needs syntax error trapping, unicode encoding support
        code = compile(self, self.source.filename, *args, **kw)
        return self.source.recode(code, self.line-1)

    def eval(self, *args):
        return eval(self.compile('eval'), *args)

    def splitlines(self, *args, **kw):
        return [Line(line, self.source, self.line+offset)
            for offset, line in enumerate(str.splitlines(self, *args, **kw))]

    for m in [
        'capitalize', 'center', 'expandtabs', 'ljust', 'lower', 'lstrip',
        'replace', 'rjust', 'rstrip', 'strip', 'swapcase', 'title',
        'translate', 'upper', 'zfill', '__add__', '__radd__', '__getslice__',
        '__mod__', '__rmod__',
    ]:
        if hasattr(str, m):
            locals()[m] = (lambda f:
                lambda self,*args,**kw: Line(
                    f(self,*args,**kw), self.source, self.line
                )
            )(getattr(str,m))







def replaces(target):
    """Class decorator to indicate that this service replaces another"""

    def decorator(cls):
        if not issubclass(cls, Service):
            raise TypeError(
                "context.replaces() can only be used in a context.Service"
                " subclass"
            )
        cls.get = staticmethod(target.get)
        return cls

    from peak.util.decorators import decorate_class
    decorate_class(decorator)

    # Ensure that context.replaces() is used only once per class suite
    cdict = sys._getframe(1).f_locals

    if cdict.setdefault('get', target.get) is not target.get:
        raise ValueError(
            "replaces() must be used only once per class;"
            " there is already a value for ``get``: %r"
            % (cdict['get'],)
        )

















