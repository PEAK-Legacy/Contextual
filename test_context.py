def additional_tests():
    import doctest
    return doctest.DocFileSuite(
        'README.txt', 'Contextual.txt', 'context_tests.txt', 'context.txt',
        optionflags=doctest.ELLIPSIS,
    )

