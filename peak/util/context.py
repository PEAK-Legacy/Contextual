"""Dynamic, contextual variables and services"""

__all__ = [
    'Action','Resource', 'Config','Setting','SettingConflict','NoValueFound',
    'AbstractProxy', 'ObjectProxy', 'Proxy', 'Wrapper', # proxies
    'Namespace', 'Global', 'new', 'snapshot', 'swap',   # system
    'call_with', 'with_', 'manager', 'gen_exc_info',    # PEP 343 impl.
]
    # XXX: clonef, qname, default_fallback, Symbol, NOT_FOUND, NOT_GIVEN

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



try:
    from peak.util.symbols import Symbol, NOT_GIVEN, NOT_FOUND

except ImportError:
    class Symbol(object):
    
        """Symbolic global constant"""
    
        __slots__ = ['_name', '_module']
        __name__   = property(lambda s: s._name)
        __module__ = property(lambda s: s._module)
    
        def __init__(self, symbol, moduleName):
            self.__class__._name.__set__(self,symbol)
            self.__class__._module.__set__(self,moduleName)
    
        def __reduce__(self):
            return self._name
    
        def __setattr__(self,attr,val):
            raise TypeError("Symbols are immutable")
    
        def __repr__(self):
            return self.__name__
    
        __str__ = __repr__
    
    
    NOT_GIVEN   = Symbol("NOT_GIVEN", __name__)
    NOT_FOUND   = Symbol("NOT_FOUND", __name__)











def with_(ctxmgr, func):
    """Perform PEP 343 "with" logic for Python versions <2.5

    The following examples do the same thing at runtime::

        Python 2.5+          Python 2.4
        ------------         -------------
        with x as y:         z = with_(x,f)
            z = f(y)

    This function is used to implement the ``call_with()`` decorator, but
    can also be used directly.  It's faster and more compact in the case where
    the function ``f`` already exists
    """
    ctx = ctxmgr.__context__()
    inp = ctx.__enter__()
    try:
        retval = func(inp)
    except:
        ctx.__exit__(*sys.exc_info())
        raise
    else:
        ctx.__exit__(None, None, None)
        return retval

















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

    def __context__(self):
        return self

    def __enter__(self):
        for value in self.geniter:
            return value
        else:
            raise RuntimeError("Generator didn't yield a value")

    def __exit__(self,*exc):
        try:
            old = gen_exc_info()
            _write_ctx()[gen_exc_info] = exc
            for value in self.geniter:
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


gen_exc_info = Global(
    "gen_exc_info", (None,None,None),
)

def default_fallback(config,key):
    """Look up the key in the config's parent scope, or error message"""
    try:
        return config.parent[key]
    except TypeError:
        if config.parent is None:
            raise NoValueFound(key)
        raise









class Config(object):
    current = staticmethod(lambda: None)

    def __init__(self, parent=None):
        if parent is None:
            parent = self.current()
        if parent is None:
            parent = getattr(self,'root',None)            
        self.parent = parent
        self.data = {}

    def __context__(self):
        return self.current(self)   # make self the current context

    def __getitem__(self,key):
        try:
            return self.data[key]
        except KeyError:
            fallback = getattr(key,'__config_fallback__',default_fallback)
            self[key] = value = fallback(self,key)
            return value

    def __setitem__(self,key,val):
        old = self.data.setdefault(key,val)
        if old is not val and old != val:
            raise SettingConflict(
                "a different value for %s is already defined" % qname(key)
            )

Config.root = Config()
Config.current = staticmethod(
    Global("current", Config.root, doc =
        """Get or set the current configuration scope"""
    )
)






class SettingConflict(Exception):
    """Attempt to set conflicting value in a scope"""

class NoValueFound(LookupError):
    """No value was found for the setting or resource"""

def Setting(name="setting", default=None, doc=None, module=None, scope=Config):

    _not_given = NOT_GIVEN
    _scope = scope.current

    def setting(value = _not_given):
        """A context.Setting that was defined without a docstring

        Call with zero arguments to get its value in the current scope, or with
        one argument to change the setting to the passed-in value.  Note that
        settings may only be changed once in a given scope, and then only if
        they have not been read within that scope.  Settings that are read
        without having been set inherit their value from the parent
        configuration of the current configuration.
        """
        if value is _not_given:
            return _scope()[setting]
        else:        
            _scope()[setting] = value
            #return value

    setting = clonef(setting,name,doc,module)
    if default is not NOT_GIVEN:
        Config.root[setting] = default

    setting.__clone__ = (
        lambda
            name=name, default=default, doc=doc,
            module=setting.func_globals.get('__name__'):
            Setting(name,default,doc,module)
    )

    return setting


class Action(object):

    __slots__ = 'config', 'managers', 'cache', 'status'

    current = staticmethod(
        Global("current", doc = "Get or set the current Action scope")
    )

    get_config = staticmethod(Config.current)
        
    def __init__(self, config=None):
        self.managers = []
        self.cache = {}
        self.status = {}
        if config is None:
            config = self.get_config()
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
            ob = ob.__context__
        except AttributeError:
            return ob
        ctx = ob()
        ob = ctx.__enter__()

        # don't call __exit__ unless __enter__ succeeded
        # (if there was an error, we wouldn't have gotten this far)
        self.managers.append(ctx)   
        return ob

    def __context__(self):
        return self

    def __enter__(self):
        if self.managers:
            raise RuntimeError("Action is already in use")
        self.manage(self.current(self))
        
    def __exit__(self, *exc):
        if not self.managers:
            raise RuntimeError("Action is not currently in use")

        managers = self.managers
        while managers:
            managers.pop().__exit__(*exc)  # XXX how do we handle errors?

        self.cache.clear()

    # TODO: prevent closed resource access during __exit__









def Resource(name="resource", factory=None, doc=None, module=None, scope=Action):

    _not_given = NOT_GIVEN
    _read_scope = scope.current
    _write_scope = scope.get_config
    _qname = qname

    def resource(factory = _not_given):
        """A context.Resource that was defined without a docstring

        Call with zero arguments to get the instance for the current action
        scope, or with one argument to set the resource's factory in the
        current *configuration* scope (which may not be the same configuration
        being used by the current action scope).
        """
        if factory is _not_given:
            _scope = _read_scope()
            try:
                return _scope[resource]
            except TypeError:
                if _scope is None:
                    raise RuntimeError(
                        "%s cannot be used outside of %s scope" %
                        (_qname(resource), _qname(scope))
                    )
                raise
        else:
            _write_scope()[resource] = factory
            #return factory

    resource = clonef(resource, name, doc, module)
    if factory is not NOT_GIVEN:
        Config.root[resource] = factory

    resource.__clone__ = (
        lambda name=name,factory=factory,doc=doc,
        module=resource.func_globals.get('__name__'),scope=scope:
            Resource(name,factory,doc,module,scope)
    )
    return resource

class AbstractProxy(object):
    """Delegates all operations (except __subject__ attr) to another object"""
    
    __slots__ = ()

    def __call__(self,*args,**kw):
        return self.__subject__(*args,**kw)

    def __getattribute__(self, attr, oga=object.__getattribute__):
        subject = oga(self,'__subject__')
        if attr=='__subject__':
            return subject
        return getattr(subject,attr)

    def __setattr__(self,attr,val, osa=object.__setattr__):
        if attr=='__subject__':
            osa(self,attr,val)
        else:
            setattr(self.__subject__,attr,val)
        
    def __delattr__(self,attr, oda=object.__delattr__):
        if attr=='__subject__':
            oda(self,attr)
        else:
            delattr(self.__subject__,attr)
    
    def __nonzero__(self):
        return bool(self.__subject__)

    def __getitem__(self,arg):
        return self.__subject__[arg]

    def __setitem__(self,arg,val):
        self.__subject__[arg] = val

    def __delitem__(self,arg):
        del self.__subject__[arg]

    def __getslice__(self,i,j):
        return self.__subject__[i:j]

    def __setslice__(self,i,j,val):
        self.__subject__[i:j] = val

    def __delslice__(self,i,j):
        del self.__subject__[i:j]
    
    def __contains__(self,ob):
        return ob in self.__subject__

    for name in 'repr str hash len abs complex int long float iter oct hex'.split():
        exec "def __%s__(self): return %s(self.__subject__)" % (name,name)

    for name in 'cmp','coerce','divmod':
        exec "def __%s__(self,ob): return %s(self.__subject__,ob)" % (name,name)

    for name,op in [
        ('lt','<'), ('gt','>'), ('le','<='), ('ge','>='),
        ('eq','=='), ('ne','!=')
    ]:
        exec "def __%s__(self,ob): return self.__subject__ %s ob" % (name,op)
        
    for name,op in [('neg','-'), ('pos','+'), ('invert','~')]:
        exec "def __%s__(self): return %s self.__subject__" % (name,op)
        
    for name, op in [
        ('or','|'),  ('and','&'), ('xor','^'), ('lshift','<<'), ('rshift','>>'),
        ('add','+'), ('sub','-'), ('mul','*'), ('div','/'), ('mod','%'),
        ('truediv','/'), ('floordiv','//')
    ]:
        exec (
            "def __%(name)s__(self,ob):\n"
            "    return self.__subject__ %(op)s ob\n"
            "\n"
            "def __r%(name)s__(self,ob):\n"
            "    return ob %(op)s self.__subject__\n"
            "\n"
            "def __i%(name)s__(self,ob):\n"
            "    self.__subject__ %(op)s=ob\n"
            "    return self\n"
        )  % locals()

    del name, op

    # Oddball signatures
    
    def __rdivmod__(self,ob):
        return divmod(ob,self.__subject__)

    def __pow__(self,*args):
        return pow(self.__subject__,*args)
    
    def __ipow__(self,ob):
        self.__subject__ **= ob
        return self

    def __rpow__(self,ob,*modulo):
        return pow(ob, self.__subject__, *modulo)


class ObjectProxy(AbstractProxy):
    """Delegates all operations (except __subject__ attr) to another object"""

    __slots__ = "__subject__"

    def __init__(self,subject):
        self.__subject__ = subject


def Proxy(func):
    class FunctionProxy(AbstractProxy):
        __slots__ = ()
        __subject__ = property(lambda self: func())
    return FunctionProxy()









class Wrapper(ObjectProxy):
    """Mixin to allow extra behaviors and attributes on proxy instance"""

    __slots__ = ()

    def __getattribute__(self, attr, oga=object.__getattribute__):       
        if attr.startswith('__'):
            subject = oga(self,'__subject__')
            if attr=='__subject__':
                return subject
            return getattr(subject,attr)
        return oga(self,attr)

    def __getattr__(self,attr):
        return getattr(self.__subject__,attr)

    def __setattr__(self,attr,val, osa=object.__setattr__):
        if (
            attr=='__subject__'
            or hasattr(type(self),attr) and not attr.startswith('__')
        ):
            osa(self,attr,val)
        else:
            setattr(self.__subject__,attr,val)
        
    def __delattr__(self,attr, oda=object.__delattr__):
        if (
            attr=='__subject__'
            or hasattr(type(self),attr) and not attr.startswith('__')
        ):
            oda(self,attr)
        else:
            delattr(self.__subject__,attr)








class Namespace(Wrapper):

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




