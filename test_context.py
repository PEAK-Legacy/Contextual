def additional_tests():
    import doctest
    return doctest.DocFileSuite(
        'context.txt', optionflags=doctest.ELLIPSIS,
    )

