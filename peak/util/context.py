"""Dynamic, contextual variables and services"""

__all__ = [
    'Action','Resource', 'Config','Setting','SettingConflict','NoValueFound',
    'Namespace', 'Global', 'new', 'snapshot', 'swap',   # system
    'Proxy', 'call_with', 'with_', 'manager', 'gen_exc_info',    # PEP 343 impl.
]
    # XXX: clonef, qname, default_fallback, Replaceable, Scope, Globals, replace

from peak.util.symbols import Symbol, NOT_GIVEN, NOT_FOUND
from peak.util.proxies import ObjectWrapper, CallbackProxy

try:
    from thread import get_ident
except ImportError:
    from dummy_thread import get_ident

import sys
from new import function
contexts = {}


def _read_ctx():
    tid = get_ident()
    try:
        return contexts[tid]
    except KeyError:
        d = contexts[tid] = {}
        return d

def _write_ctx():
    tid = get_ident()
    try:
        d = contexts[tid]
    except KeyError:
        d = contexts[tid] = {}
    if '__frozen__' in d:
        d = contexts[tid] = d.copy()
        del d['__frozen__']
    return d

def with_(ctx, func):
    """Perform PEP 343 "with" logic for Python versions <2.5

    The following examples do the same thing at runtime::

        Python 2.5+          Python 2.4
        ------------         -------------
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
    typ,val,tb = gen_exc_info()
    if typ:
        try:
            raise typ,val,tb
        finally:
            del typ,val,tb

def replaces(target):
    def decorator(cls):
        assert issubclass(cls,Replaceable)
        cls.current = staticmethod(target.current)
        return cls
    from peak.util.decorators import decorate_class
    decorate_class(decorator)

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

        print with_(x,do_it)
    """
    return with_.__get__(ctxmgr, type(ctxmgr))





class _GeneratorContextManager(object):
    """Helper class for ``@context.manager``"""

    __slots__ = "geniter"

    def __init__(self, geniter):
        self.geniter = geniter

    def __enter__(self):
        for value in self.geniter:
            return value
        else:
            raise RuntimeError("Generator didn't yield a value")

    def __exit__(self,*exc):
        try:
            old = gen_exc_info()
            _write_ctx()[gen_exc_info] = exc
            try:
                for value in self.geniter:
                    break
                else:
                    return True     # generator swallowed exception
            except:
                if not exc or sys.exc_info()[1] is not exc[1]: raise
                return False
            raise RuntimeError("Generator didn't stop")
        finally:
            _write_ctx()[gen_exc_info] = old


def manager(func):
    """Emulate 2.5 ``@contextmanager`` decorator"""
    def helper(*args, **kwds):
        return _GeneratorContextManager(func(*args, **kwds))
    helper.__name__ = func.__name__
    helper.__doc__  = func.__doc__
    return helper



def make_variable(name, factory, doc, module, scope):
    _not_given = NOT_GIVEN
    _qname = qname
    read_scope = scope.get_read_scope_getter()
    write_scope = scope.get_write_scope_getter()

    def var(value = _not_given):
        if value is _not_given:
            _scope = read_scope()
            if _scope is None:
                raise RuntimeError(
                    "%s cannot be used outside of %s scope" %
                    (_qname(var), _qname(scope))
                )
            try:
                return _scope[var]
            except KeyError:
                value = factory()
                write_scope()[var] = value
                return value

        write_scope()[var] = value

    var = clonef(var,name,doc,module,level=3)
    var.__clone__ = lambda name, factory=factory: make_variable(
        name, factory, doc, module, scope
    )
    return var

def clonef(f, name, doc, module, frame=None, level=2):
    """Clone a func with reset globals to look like it came from elsewhere"""
    if module is not None:
        __import__(module)
        _globals = sys.modules[module].__dict__
    else:
        _globals = (frame or sys._getframe(level)).f_globals
    # clone the function using the targeted module (or caller's) globals
    f = function(f.func_code, _globals, name, f.func_defaults, f.func_closure)
    f.__doc__  = doc  or f.__doc__
    return f

def qname(f):
    if hasattr(f,'__module__'):
        m = f.__module__
    else:
        m = f.func_globals.get('__name__')
    if m:
        return '%s.%s' % (m,f.__name__)
    return f.__name__

@manager
def _pusher(f,val):
    """Helper that sets and resets a context global"""
    old = f()
    _write_ctx()[f] = val
    yield None  # XXX should this be old, or val instead?
    _write_ctx()[f] = old
    reraise()

def default_fallback(config,key):
    """Look up the key in the config's parent scope, or error message"""
    try:
        return config.parent[key]
    except TypeError:
        if config.parent is None:
            raise NoValueFound(key)
        raise


_ctx_stack = object()

def delegated_enter(self):
    ctx = self.__context__()
    _write_ctx().setdefault(_ctx_stack,[]).append(ctx)
    return ctx.__enter__()

def delegated_exit(self, typ, val, tb):
    ctx = _read_ctx()[_ctx_stack].pop()
    return ctx.__exit__(typ, val, tb)



class ScopedClass(type):
    def __init__(cls, name, bases, cdict):
        super(ScopedClass, cls).__init__(cls,name,bases,cdict)
        if cls.scope is not None and 'current' not in cdict:
            cls.current = staticmethod(
                make_variable(
                    name+".current", cls.__default__,
                    "Get or set the current "+name+" scope",
                    cls.__module__, cls.scope
                )
            )


class Globals(object):

    @classmethod
    def get_read_scope_getter(cls):
        return _read_ctx

    @classmethod
    def get_write_scope_getter(cls):
        return _write_ctx

    def __new__(*args):
        raise TypeError("Globals can't be instantiated")


@manager
def replace(func, val):
    """Context that temporarily replaces a scope, variable, or proxy"""
    if type(func).__name__=='FunctionProxy' and hasattr(func,'__func__'):
        func = func.__func__
    elif isinstance(func, ScopedClass) and hasattr(func, 'current'):
        func = func.current
    old = func()
    func(val)
    yield val
    func(old)
    reraise()


class Replaceable(object):

    __slots__ = ()
    __metaclass__ = ScopedClass

    scope = Globals

    @classmethod
    def __default__(cls):
        return cls()

    def __context__(self):
        return replace(self.current, self)

    __enter__ = delegated_enter
    __exit__  = delegated_exit

    @classmethod
    def proxy(cls):
        return Proxy(cls.current)


class Scope(Replaceable):
    __slots__ = ()

    @classmethod
    def get_read_scope_getter(cls):
        return cls.current

    @classmethod
    def get_write_scope_getter(cls):
        return cls.current









class Config(Scope):
    __slots__ = 'parent', 'data'

    def __init__(self, parent=NOT_GIVEN):
        if parent is NOT_GIVEN:
            parent = self.current()
        self.parent = parent
        self.data = {}

    @classmethod
    def __default__(cls):
        return cls.root     # default Config is the root config

    def __getitem__(self,key):
        try:
            return self.data[key]
        except KeyError:
            fallback = getattr(key, '__config_fallback__', default_fallback)
            self[key] = value = fallback(self,key)
            return value

    def __setitem__(self,key,val):
        old = self.data.setdefault(key,val)
        if old is not val and old != val:
            raise SettingConflict(
                "a different value for %s is already defined" % qname(key)
            )

Config.root = Config(None)












class Action(Scope):
    __slots__ = 'config', 'managers', 'cache', 'status'

    @classmethod
    def get_write_scope_getter(cls):
        return Config.current

    @classmethod
    def __default__(cls):
        return None     # no default Action

    def __init__(self, config=None):
        self.managers = []
        self.cache = {}
        self.status = {}
        if config is None:
            config = self.get_write_scope_getter()()
        self.config = config

    def __getitem__(self,key):
        try:
            res = self.cache[key]
        except KeyError:
            cfg = self.config
            factory = cfg[key]
            status = self.status.get(key)
            if status:
                raise RuntimeError(
                    "Circular dependency for %s (via %s)"
                    % (qname(key),factory)
                )
            self.status[key] = 1    # recursion guard
            try:
                res = self.cache[key] = self.manage(
                    with_(cfg, lambda arg: factory())
                )
            finally:
                del self.status[key]
        return res


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

    def __enter__(self):
        if self.managers:
            raise RuntimeError("Action is already in use")
        self.manage(replace(self.current, self))

    def __exit__(self, *exc):
        if not self.managers:
            raise RuntimeError("Action is not currently in use")

        managers = self.managers
        while managers:
            managers.pop().__exit__(*exc)  # XXX how do we handle errors?

        self.cache.clear()

    # TODO: prevent closed resource access during __exit__












class SettingConflict(Exception):
    """Attempt to set conflicting value in a scope"""

class NoValueFound(LookupError):
    """No value was found for the setting or resource"""

def Setting(name="setting", default=None, doc=None, module=None, scope=Config):
    doc = doc or \
        """A context.Setting that was defined without a docstring

        Call with zero arguments to get its value in the current scope, or with
        one argument to change the setting to the passed-in value.  Note that
        settings may only be changed once in a given scope, and then only if
        they have not been read within that scope.  Settings that are read
        without having been set inherit their value from the parent
        configuration of the current configuration.
        """
    setting = make_variable(name, lambda:default, doc, module, scope)
    if default is not NOT_GIVEN:
        Config.root[setting] = default
    return setting

def Resource(name="resource", factory=None, doc=None, module=None, scope=Action):
    doc = doc or \
        """A context.Resource that was defined without a docstring

        Call with zero arguments to get the instance for the current action
        scope, or with one argument to set the resource's factory in the
        current *configuration* scope (which may not be the same configuration
        being used by the current action scope).
        """
    resource = make_variable(name, lambda:factory, doc, module, scope)

    if factory is not NOT_GIVEN:
        Config.root[resource] = factory

    return resource




class Namespace(ObjectWrapper):

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            if key.startswith('func_') or key.startswith('__'):
                return getattr(self.__subject__,key)
            return self[key]

    def __getitem__(self,key):
        if '.' in key:
            for key in key.split('.'):
                self = self[key]
            return self
        try:
            return self.__dict__[key]
        except KeyError:
            # TODO: verify syntax of key: nonempty, valid chars, ...?
            val = self.__clone__("%s.%s" % (self.__name__,key), NOT_GIVEN)
            if key=='*':
                if hasattr(self,'__namespace__'):
                    ns = self.__namespace__
                    val.__config_fallback__ = lambda m,k: m[ns['*']]
                else:
                    val.__config_fallback__ = lambda m,k: default_fallback
            else:
                val.__config_fallback__ = lambda m,k: m[self['*']](m,k)

            val.__namespace__ = self
            self.__dict__[key] = val = Namespace(val)
            return val

    def __contains__(self,key):
        for key in key.split('.'):
            if key not in self.__dict__:
                return False
            self = self[key]
        return True


    def __iter__(self):
        for key in self.__dict__:
            if not key.startswith('__') and key!='*':
                yield key


def Global(name="unnamed_global", default=None, doc=None, module=None):

    _not_given = NOT_GIVEN
    __read = _read_ctx
    __pusher = _pusher

    def f(value = _not_given):
        """A context.Global that was defined without a docstring

        Call with zero arguments to get its value, or with one argument to
        receive a contextmanager that temporarily sets the global to the
        passed-in value, for the duration of the "with:" block it's used to
        create.
        """
        if value is _not_given:
            return __read().get(f,default)
        else:
            return __pusher(f,value)

    f = clonef(f,name,doc,module)   # *must* rebind f to cloned function here
    f.__clone__ = (
        lambda name=name,default=default,doc=doc,
        module=f.func_globals.get('__name__'): Global(name,default,doc,module))
    return f

gen_exc_info = Global("gen_exc_info", (None,None,None))

Proxy = CallbackProxy







def new():
    """Return a new, empty thread state"""
    return {'__frozen__':True}

def snapshot():
    """Return a snapshot of the all of this thread's current state globals

    This returns an object that can be passed into ``context.swap()`` to
    restore all context globals to what they were at this point in time.
    A copy-on-write strategy is used, so that snapshots can always be taken
    in constant time, and no extra memory is used if multiple snapshots are
    taken during a period where no changes have occurred.
    """
    ctx = _read_ctx()
    ctx['__frozen__'] = True
    return ctx

def swap(state):
    """Set this thread's state to `snapshot`, returning a "before" snapshot

    `snapshot` should be a value previously returned by either the
    ``snapshot()`` function or by ``swap()``.  Before the thread state is set,
    a new snapshot is taken and returned.

    The passed-in snapshot is not modified by this routine, so you can reuse
    it as many times as you want to restore that particular state.
    """
    old = snapshot()
    contexts[get_ident()] = state
    return old


def doctest_suite():
    import doctest
    return doctest.DocFileSuite(
        'context.txt', optionflags=doctest.ELLIPSIS, module_relative=False,
    )




