# Note:
#   the setup.py build target does *NOT* update the _wf.so extension.
#   to do so, use the develop target (that it is why it appears first).

develop:

	python setup.py develop



# The minus sign means ignore any errors (e.g. if there is nothing to uninstall)
install:
	-pip uninstall ncpyview
	python setup.py install --verbose

uninstall:
	-pip uninstall ncpyview
	(cd ~/.local/bin; \rm -f ncpyview)

clean:
	\rm -rf build _*.so
	find . -name __pycache__ | xargs \rm -rf
	find . -name *.pyc | xargs \rm -f
	find . -name *.so | xargs \rm -f
	python setup.py clean

tests:
	nosetests -w tests 

tags:
	ctags -R -f .tags

.PHONY: clean tests
