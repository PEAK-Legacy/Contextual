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
   the fear of later disasters in retribution for our sins),

2. attempt to use a dependency injection framework, paying extra now to be
   reassured that things will work out later, or

3. use a thread-local variable, and bear the cost of introducing a possible
   threading dependency, and still not having a reasonable way to test or
   configure alternate implementations.  Plus, thread-locals don't really
   support asynchronous programming or co-operative multitasking.  What if
   somebody wants to use your library under Twisted, and needs private
   instances for each socket connection?

But now there's a better choice.

The "Contextual" library (``peak.context``) lets you create pseudo-singletons
and pseudo-global variables that are context-sensitive and easily replaceable.
They look and feel just like old-fashioned globals and singletons, but because
they are safely scalable to threads and tasks (as well as being replaceable for
testing or other dynamic contexts), you don't have to worry about what happens
"later".

Contextual singletons are even better than thread-local variables, because they
support asynchronous programming with microthreads, coroutines, or frameworks
like Twisted.  A simple context-switching API lets you instantly swap out all
the services and variables from one logical task, with those of another task.
This just isn't possible with ordinary thread-locals.

Meanwhile, "client" code that uses context-sensitive objects remains unchanged:
the code simply uses whatever the "current" object is supposed to be.

And isn't that all you wanted to do in the first place?


Replaceable Singletons
----------------------

Here's what a simple "global" counter service implemented with ``peak.context``
looks like::

    >>> from peak import context

    >>> class Counter(context.Service):
    ...     value = 0
    ...
    ...     def inc(self):
    ...         self.value += 1
    ...

    >>> Counter.value
    0
    >>> Counter.inc()
    >>> Counter.value
    1

Code that wants to use this global counter just calls ``Counter.inc()`` or
accesses ``Counter.value``, and it will automatically use the right ``Counter``
instance for the current thread or task.  Want to use a fresh counter for
a test?  Just do this::

    with Counter.new():
        # code that uses the standard count.* API

Within the ``with`` block, any code that refers to ``count`` will be using the
new ``Counter`` instance you provide.  If you need to support Python 2.4, the
``context`` library also includes a decorator that emulates a ``with``
statement::

    >>> Counter.value     # before using a different counter
    1

    >>> @context.call_with(Counter.new())
    ... def do_it(c):
    ...     print Counter.value
    0

    >>> Counter.value     # The original counter is now in use again
    1

The ``@call_with`` decorator is a bit uglier than a ``with`` statement, but
it works about as well.  You can also use an old-fashioned try-finally block,
or some other before-and-after mechanism like the ``setUp()`` and
``tearDown()`` methods of a test to replace and restore the active instance.


Pluggable Services
------------------

Want to create an alternative implementation of the same service, that can
be plugged in to replace it?  That's simple too::

    >>> class DoubleCounter(context.Service):
    ...     context.replaces(Counter)
    ...     value = 0
    ...     def inc(self):
    ...         self.value += 2

To use it, just do::

    with DoubleCounter.new():
        # code in this block that calls ``Counter.inc()`` will be incrementing
        # a ``DoubleCounter`` instance by 2

Or, in Python 2.4, you can do something like::

    >>> @context.call_with(DoubleCounter.new())
    ... def do_it(c):
    ...     print Counter.value
    ...     Counter.inc()
    ...     print Counter.value
    0
    2

And of course, once a replacement is no longer in use, the original instance
becomes active again::

    >>> Counter.value
    1

All this, with no interfaces to declare or register, and no XML or
configuration files to write.  However, if you *want* to use configuration
files to select implementations of global services, you can still have them:
setting ``Counter <<= DoubleCounter`` will set the current ``Counter`` factory
to ``DoubleCounter``, so you can just have a configuration file loader set up
whatever services you want.  You can even take a snapshot of the entire current
context and restore all the previous values::

    with context.empty():
        # code to read config file and set up services
        # code that uses the configured services

This code won't share any services with the code that calls it; it will not
only get its own private ``Counter`` instance, but a private instance of any
other ``Service`` objects it uses as well.  (Instances are created lazily
in new contexts, so if you don't use a particular service, it's never created.)
Try doing that with global or thread-local variables!

In addition to these simple pseudo-global objects, ``peak.context`` also
supports other kinds of context-sensitivity, like the concept of "settings"
in a "current configuration" and the concept of "resources" in a "current
action" (that are notified whether the action completed successfully or exited
with an error).  These features are orders of magnitude simpler in their
implementation and use than the corresponding features in the earlier
``peak.config`` and ``peak.storage`` frameworks, but provide equivalent or
better functionality.

For more details, please consult the Contextual developer's guide.


TODO
----

0.7
 * Finish the developer's guide!
 
 * Configuration files

 * Components w/state binding and ``**kw`` attrs update on ``__init__`` and
   ``.new()``

0.8
 * State ``__enter__`` should lock the state to the current thread, w/o
   ``__exit__`` or ``swap()`` or on_exit being possible from other threads,
   so that they will be thread-safe.

 * Detect value calculation cycles

 * Resource pooling/caching


STATUS
------

This package is in active development, but not all features are stable and
documented.  ``Service`` objects work as advertised, as does the support for
using "with"-like operations in older versions of Python.  Most of the other
features haven't been used (or even documented!) in any real way yet, and so
the designs are still subject to change prior to an actual 0.7a1 release.

(All the included code is covered by tests, though, so you can always dig
through them for technical documentation; the developer guide and tutorial
is just woefully incomplete as yet.)

Source distribution snapshots of Contextual are generated daily, but you can
also update directly from the `development version`_ in SVN.

.. _development version: svn://svn.eby-sarna.com/svnroot/Contextual#egg=Contextual-dev

