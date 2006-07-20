=========================================================
Making the World Safe for "Globals" with ``peak.context``
=========================================================

So you're writing a library, and you have this object that keeps showing up
in parameters or attributes everywhere, even though there's only ever *one*
of that thing at a given moment in time.  Should you use a global variable or
singleton?

Most of us know we "shouldn't" use globals, and some of us know that singletons
are just another kind of global!  But there are times when they both just seem
so darn attractive.  They're *so* easy to create and use, even though they're
also the bane of testability, maintainability, configurability,
thread-safety...  Heck, you can pretty much name it, and it's a problem with
globals and singletons.

Programming pundits talk about using "dependency injection" or "inversion of
control" (IoC) to get rid of global variables.  And there are many dependency
injection frameworks for Python (including Zope 3 and ``peak.config``).

The problem is, these frameworks typically require you to declare interfaces,
register services, create XML configuration files, and/or ensure that every
object in your application knows where to look up services -- replacing one
"globals" problem with another!  Not only does all this make things more
complex than they need to be, it disrupts your programming flow by making you
do busywork that doesn't provide any new benefits to your application.

So, most of us end up stuck between various unpalatable choices:

1. use a global and get it over with (but suffer a guilty conscience and
   the fear of later disasters in retribution for our sins), or

2. attempt to use a dependency injection framework, paying extra now to be
   reassured that things will work out later.

3. use a thread-local variable, and bear the cost of introducing a possible
   threading dependency, and still not having a reasonable way to test or
   configure alternate implementations.  Plus, thread-locals don't really
   support asynchronous programming or co-operative multitasking.  What if
   somebody wants to use your library under Twisted?

But now there's a better choice.

The "Contextual" library (``peak.context``) lets you create pseudo-singletons
and pseudo-global variables that are context-sensitive and easily replaceable.
They look and feel just like old-fashioned globals and singletons, but because
they are safely scalable and replaceable, you don't have to worry about what
happens "later".

Contextual singletons are even better than thread-local variables, because they
support asynchronous programming with microthreads, coroutines, or frameworks
like Twisted.  A simple context-switching API lets you instantly change from
one logical task's context to another.  This just isn't possible with ordinary
thread-locals.  Meanwhile, "client" code that uses context-sensitive objects
remains unchanged: the code simply uses whatever the "current" object is
supposed to be.

And, isn't that all you wanted to do in the first place?


Replaceable Singletons
----------------------

Here's what a simple "global" counter object implemented with ``peak.context``
looks like::

    >>> from peak.util import context

    >>> class Counter(context.Replaceable):
    ...     value = 0
    ...
    ...     def inc(self):
    ...         self.value += 1
    ...

    >>> count = Counter.proxy()

    >>> count.value
    0
    >>> count.inc()
    >>> count.value
    1

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

    >>> count.value     # before using a different counter
    1

    >>> @context.call_with(Counter())
    ... def do_it(c):
    ...     print count.value
    0

    >>> count.value     # The original counter is now in use again
    1

The ``@call_with`` decorator is a bit uglier than a ``with`` statement, but
it works about as well.  You can also use an old-fashioned try-finally block,
or some other before-and-after mechanism like the ``setUp()`` and
``tearDown()`` methods of a test to replace and restore the active instance.


Pluggable Services
------------------

Want to create an alternative implementation of the same service, that can
be plugged in to replace it?  That's simple too::

    >>> class DoubleCounter(context.Replaceable):
    ...     context.replaces(Counter)
    ...     value = 0
    ...     def inc(self):
    ...         self.value += 2
    ...

To use it, just do::

    with DoubleCounter():
        # code in this block that calls ``count.inc()`` will be incrementing
        # a ``DoubleCounter`` instance by 2

Or, in Python 2.4, you can do something like::

    >>> @context.call_with(DoubleCounter())
    ... def do_it(c):
    ...     print count.value
    ...     count.inc()
    ...     print count.value
    0
    2

And of course, once a replacement is no longer in use, the original instance
becomes active again::

    >>> count.value
    1

All this, with no interfaces to declare or register, and no XML or
configuration files to write.  However, if you *want* to use configuration
files to select implementations of global services, you can still have them:
calling ``Counter.current(foo)`` will set the current ``Counter`` to ``foo``,
so you can just have a configuration file loader set up whatever services you
want.  You can even take a snapshot of the entire current context and restore
all the previous values::

    old = context.swap(context.new())
    try:
        # code to read config file and set ``current()`` services
        # code that uses the configured services
    finally:
        context.swap(old)   # restore the previous context

This code won't share any "globals" with the code that calls it; it will not
only get its own private ``Counter`` instance, but a private instance of any
other ``Replaceable`` objects it uses as well.  (Instances are created lazily
in new contexts, so if you don't use a particular service, it's never created.)
Try doing that with global or thread-local variables!

In addition to these simple pseudo-global objects, ``peak.context`` also
supports other kinds of context-sensitivity, like the concept of "settings"
in a "current configuration" and the concept of "resources" in a "current
action" (that are notified whether the action completed successfully or exited
with an error).  These features are orders of magnitude simpler in their
implementation and use, than the corresponding features in the earlier
``peak.config`` and ``peak.storage`` frameworks, but provide equivalent or
better functionality.

For more details, please consult the Contextual reference manual.

