
if [  "$#" -eq 1 ]
then
    # Usage: <script> path_to_tests
	nosetests -c nosetests.cfg -a '!slow' $*
else
	# Usage: <script> attribute_expression path_to_tests
	# e.g. <script> '!slow' alber.tests
	nosetests -c nosetests.cfg -a $*
fi
