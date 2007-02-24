import sys
from thread import get_ident
from new import function
from peak.util.symbols import NOT_GIVEN
from peak.util.proxies import ObjectWrapper

__all__ = [
    'Service', 'replaces', 'Config', 'setting', 'SettingConflict',
    'Action', 'resource', 'namespace', 'App', 'parameter',
    'Delegated', 'manager', 'reraise', 'with_', 'call_with',
]


def _clonef(src, impl, name=None):
    """Create a copy of function `impl` that looks like `src`"""
    f = function(
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

class SettingConflict(Exception):
    """Attempt to set conflicting value in a scope"""

class NoValueFound(LookupError):
    """No value was found for the setting or resource"""




_params = {}

def _get_param(key):
    return _params[get_ident()][key]

def _set_param(key, value):
    ctx = _params.setdefault(get_ident(),{})
    if value is NOT_GIVEN:
        if key in ctx:
            del ctx[key]
    else:
        ctx[key] = value

class _ParamContext(object):
    """Context manager that temporarily changes a parameter value"""

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __enter__(self):
        try:
            self.old = _get_param(self.key)
        except KeyError:
            self.old = NOT_GIVEN            
        _set_param(self.key, self.value)
        return self.value

    def __exit__(self,typ,val,tb):
        _set_param(self.key, self.old)











def parameter(func):
    """Decorator to create a dynamic parameter object from a function
    """
    _gp = _get_param
    _sp = _set_param
    _ng = NOT_GIVEN
    _pc = _ParamContext

    def impl(value=NOT_GIVEN):
        if value is _ng: 
            try:
                return _gp(func)
            except KeyError:
                value = func()
                _sp(func, value)
                return value
        return _pc(func, value)

    return _clonef(func, impl)


_delegates = parameter(lambda: [])

class Delegated(object):
    """Allow replacing ``__enter__`` and ``__exit__`` w/a ``__context__()``"""

    __slots__ = ()  # pure mixin class

    def __enter__(self):
        mgr = self.__context__()
        _delegates().append((self,mgr))
        return mgr.__enter__()

    def __exit__(self, typ, val, tb):
        ctx, mgr = _delegates().pop()
        assert ctx is self, "context stack is corrupted"
        return mgr.__exit__(typ, val, tb)




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
                cm = _exc_info((typ,value,traceback)); cm.__enter__()
                try:
                    self.gen.next()
                finally:
                    cm.__exit__(None, None, None)                       
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

_exc_info = parameter(lambda: (None, None, None))

def manager(func):
    """Emulate 2.5 ``@contextli.contextmanager`` decorator"""
    gcm = _GeneratorContextManager
    return _clonef(func, lambda *args, **kwds: gcm(func(*args, **kwds)))
  
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
    typ,val,tb = _exc_info()
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





def get(cls):
    """Get the "current instance" of this service class"""
    try:
        return _get_param(cls)
    except KeyError:
        _set_param(cls, cls.__default__())
        return _get_param(cls)


def redirect_attribute(cls, name, payload):
    meta = type(cls)
    if getattr(meta, '__for_class__', None) is not cls:
        cls.__class__ = meta = type(meta)(
            cls.__name__+'Class', (meta,), {'__module__':cls.__module__, '__for_class__':cls}
        )
        # XXX activate_attrs(meta)?

    f = payload
    if hasattr(f,'__call__'):
        f = _clonef(
            payload, lambda *args,**kw: getattr(cls.get(), name)(*args,**kw)
        )
    setattr(meta, name, property(lambda s: getattr(s.get(), name)))

_ignore = {
    '__name__':1, '__module__':1, '__return__':1, '__slots__':1, 'get':1,
    '__init__':1, '__metaclass__':1, '__doc__':1, '__call__': 1, '__new__':1, 
}.get













class ServiceClass(type):

    def __new__(meta, name, bases, cdict):
        cls = super(ServiceClass, meta).__new__(meta, name, bases, cdict)
        if 'get' not in cdict:
            cls.get = staticmethod(classmethod(get).__get__(None, cls))
        for k, v in cdict.items():
            if not isinstance(v, (classmethod,staticmethod))and not _ignore(k):
                redirect_attribute(cls, k, v)
        return cls


class Service(Delegated):
    """A replaceable, thread-local singleton"""

    __slots__ = ()  # pure mixin class
    __metaclass__ = ServiceClass

    def __context__(self):
        return _ParamContext(self.get.im_self, self)

    def __default__(cls):
        return with_(App.get().config, lambda cfg: cfg[cls]())

    __default__ = classmethod(__default__)
    
    def __config_fallback__(cls, scope, key):
        if scope.parent is None:
            return cls
        return scope.parent[key]

    __config_fallback__ = classmethod(__config_fallback__)









class Config(Service):
    """A write-once, read-many mapping service with inheritance"""

    __slots__ = 'parent', 'data'

    def __init__(self, parent=NOT_GIVEN):
        if parent is NOT_GIVEN:
            parent = self.get()
        self.parent = parent
        self.data = {}

    def __default__(cls):
        return cls.root

    __default__ = classmethod(__default__)

    def __getitem__(self,key):
        try:
            return self.data[key]
        except KeyError:
            fallback = key.__config_fallback__ #getattr(, default_fallback)
            self[key] = value = fallback(self,key)
            return value


    def __setitem__(self, key, val):
        old = self.data.setdefault(key, val)
        if old is not val and old != val:
            raise SettingConflict(
                "a different value for %s is already defined" % (key,) #qname
            )

Config.root = Config(None)








def setting(func):
    """Decorator to create a configuration setting from a function
    """
    return make_var(func, Config)

def resource(func):
    """Decorator to create a configuration setting from a function
    """
    return make_var(_clonef(func, lambda s,k: func), Action)


def make_var(func, scope, name=None):

    _get = scope.get

    def impl():
        return _get()[impl]

    def fallback(scope, key):
        parent = scope.parent
        if parent is None:
            return func(scope, key)
        return parent[key]
        
    impl = _clonef(func, impl, name)
    impl.__config_fallback__ = fallback
    impl.__scope__ = scope

    return impl


def default_fallback(scope, key):
    """Fallback used for a namespace's top-level settings' wildcard results"""
    parent = scope.parent
    if parent is None:
        raise KeyError(key)
    return parent[key]




class Action(Service):
    """Service for managing transaction-scope resources"""

    def __init__(self, config=None):
        self.managers = []
        self.cache = {}
        self.status = {}
        if config is None:
            config = Config.get()
        self.config = config

    def __default__(cls):
        raise RuntimeError("No Action is currently active")

    __default__ = classmethod(__default__)   

    def __enter__(self):
        if self.managers:
            raise RuntimeError("Action is already in use")
        self.manage(None, _ParamContext(self.get.im_self, self))

    def __exit__(self, *exc):
        if not self.managers:
            raise RuntimeError("Action is not currently in use")

        managers = self.managers
        cache = self.cache

        while managers:
            key, ctx = managers.pop()
            ctx.__exit__(*exc)  # XXX how do we handle errors?
            #if key in cache:
            #    del cache[key]
            #    status[key] = -1
        self.cache.clear()






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
                    key, with_(cfg, lambda arg: factory())
                )
            finally:
                del self.status[key]
        return res

    def manage(self, key, ob):
        try:
            enter = ob.__enter__
        except AttributeError:
            return ob
        ctx = ob
        ob = ctx.__enter__()

        # don't call __exit__ unless __enter__ succeeded
        # (if there was an error, we wouldn't have gotten this far)
        self.managers.append((key,ctx))
        return ob






class _AppSwapper(object):
    __slots__ = 'old', 'new'

    def __init__(self, new):
        self.new = new

    def __enter__(self):
        self.old = App.get().params
        new = self.new
        _params[get_ident()] = new.params
        return new

    def __exit__(self, typ, val, tb):
        _params[get_ident()] = self.old



























class App(object):
    """Top-level scope for all parameters and services"""

    __metaclass__ = Service.__class__

    def __init__(self, config=NOT_GIVEN):
        self.params = {App: self}
        if config==NOT_GIVEN:
            config = Config.get()
        self.config = config
        
    def __default__(cls):
        new = cls()
        new.params = _params.setdefault(get_ident(), new.params)
        return new

    __default__ = classmethod(__default__)

    def swap(self):
        return _AppSwapper(self)

    '''def copy(self):
        """Return a new App based on the same config"""
        return type(self)(self.config)

    def clone(self):
        new = self.copy()
        npsd = new.params.setdefault
        for k,v in self.params.iteritems():
            npsd(k, v)
        return new'''










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
            impl = make_var(me, me.__scope__, "%s.%s" % (self.__name__, key))

            if key=='*':
                if hasattr(me, '__namespace__'):
                    ns = me.__namespace__
                    impl.__config_fallback__ = lambda m,k: m[ns['*']]
                else:
                    impl.__config_fallback__ = lambda m,k: default_fallback
            else:
                impl.__config_fallback__ = lambda m,k: m[self['*']](m,k)
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
    if 'get' in cdict:
        print cdict, target.get
    if cdict.setdefault('get', target.get) is not target.get:
        raise ValueError(
            "replaces() must be used only once per class;"
            " there is already a value for ``get``: %r"
            % (cdict['get'],)
        )
















