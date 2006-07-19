=========================================================
Making the World Safe for "Globals" with ``peak.context``
=========================================================

Global variables and singletons are an attractive nuisance.  Seductively easy
to create and use, but the bane of testability, maintainability,
configurability, thread-safety...  the problems of using simple globals go on
and on and on.

The "Contextual" library (``peak.context``) solves this problem by allowing you
to create pseudo-global objects and variables that are context-sensitive
and easily replaceable.

While there are many "dependency injection" frameworks for Python (including
Zope 3 and ``peak.config``), the "Contextual" library doesn't require you to
declare interfaces, register services, create XML configuration files, or
ensure that every object in your application knows where to look up services.

Instead, ``peak.context`` focuses on making globals *safe* and *replaceable*,
while retaining the ease-of-definition and ease-of-use of simple global
variables or singletons.  And, unlike thread-local variables, ``peak.context``
supports asynchronous programming with microthreads, coroutines, or frameworks
like Twisted.  A simple context-switching API lets you instantly change from
one logical task's context to another.  This just isn't possible with ordinary
thread-locals.

Here's what a simple "global" counter object looks like::

    from peak import context

    class Counter(context.Replaceable):
        value = 0

        def inc(self):
            self.value += 1

    count = Counter.proxy()

Code that wants to use this global counter just calls ``count.inc()`` or
accesses ``count.value``, and it will automatically use the right ``Counter``
instance for the current thread or task.  Want to use a fresh counter for
a test?  Just do this::

    with Counter():
        # code that uses the standard count.* API

Within the ``with`` block, any code that refers to ``count`` will be using the
new ``Counter`` instance you provide.  If you need to support Python 2.4, the
``context`` library also includes a decorator that emulates a ``with``
statement::

    @context.call_with(Counter())
    def do_it(c):
        # code that uses the standard count.* API

It's a bit uglier, but it works just as well.  You can also use an old-
fashioned try-finally block, or some other before-and-after mechanism like
the ``setUp()`` and ``tearDown()`` methods of a test to replace and restore
the active instance.

Want to create an alternate implementation?  That's simple too::

    class DoubleCounter(context.Replaceable):
        current = Counter.current
        value = 0
        def inc(self):
            self.value += 2

    with DoubleCounter():
        # code in this block that calls ``count.inc()`` will be incrementing
        # a ``DoubleCounter`` instance by 2

There are no interfaces to declare or register, and no XML or configuration
files to write.  However, if you *want* to use configuration files to select
implementations of global services, you can still have them: calling
``Counter.current(foo)`` will set the current ``Counter`` to ``foo``, so you
can just have a configuration file loader set up whatever services you want.
You can even take a snapshot of the entire current context and restore all the
previous values::

    old = context.swap(context.new())
    try:
        # code to read config file and set ``current()`` services
        # code that uses the configured services
    finally:
        context.swap(old)   # restore the previous context

This code won't share any "globals" with the code that calls it; it will get
its own private ``Counter`` instance, for example.

In addition to these simple pseudo-global objects, ``peak.context`` also
supports other kinds of context-sensitivity, like the concept of "settings"
in a "current configuration" and the concept of "resources" in a "current
action", that are notified whether the action completed successfully or exited
with an error.  These features are orders of magnitude simpler in their
implementation and use, than the corresponding features in the earlier
``peak.config`` and ``peak.storage`` frameworks.

For more details, please consult the Contextual reference manuals.

