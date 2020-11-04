new_string = 'This is a String' # storing a string
print('ID:', id(new_string)) # shows the object identifier (address)
print('Type:', type(new_string)) # shows the object type
print('Value:', new_string) # shows the object value

# simple string
simple_string = 'hello' + " I'm a simple string"
print(simple_string)

# multi-line string, note the \n (newline) escape character automatically created
multi_line_string = """Hello I'm
a multi-line
string!"""
print(multi_line_string)

# Normal string with escape sequences leading to a wrong file path!
escaped_string = "C:\the_folder\new_dir\file.txt"
print(escaped_string)  # will cause errors if we try to open a file here

# raw string keeping the backslashes in its normal form
raw_string = r'C:\the_folder\new_dir\file.txt'
print(raw_string)

# unicode string literals
string_with_unicode = u'H\u00e8llo!'
print(string_with_unicode)

# working with unicode characters
more_unicode = u'I love Pizza \U0001F355! Shall we book a cab \U0001F695 to get pizza?'
print(more_unicode)
print(string_with_unicode + '\n' + more_unicode)
' '.join([string_with_unicode, more_unicode])
print(more_unicode[::-1]) # reverses the string

# String operations
# Different ways of String concatenation
print(u'Hello \u263A' + ' and welcome ' + u'to Python \U0001F40D!')
print('Hello' ' and welcome ' 'to Python!')

# concatenation of variables and literals
s1 = u'Python \U0001F4BB!'
print('Hello ' + s1)
print(u'Hello \u263a ' + s1)

# we cannot concatenate a variable and a literal using this method
#print(u'Hello \u263a ' s1)

# some more ways of concatenating strings
s2 = u'--\U0001F40DPython\U0001F40D--'
print(s2 * 5)
print(s1 + s2)
print((s1 + s2)*3)
print('Python!--Python--Python!--Python--Python!--Python--')

# concatenating several strings together in parentheses
s3 = ('This '
      'is another way '
      'to concatenate '
      'several strings!')
print(s3)

# checking for substrings in a string
print('way' in s3)
print('python' in s3)

# computing total length of the string
print(len(s3))

# String indexing and slicing
# creating a string
s = 'PYTHON'
print(s, type(s))

# depicting string indexes
for index, character in enumerate(s):
    print('Character', character+':', 'has index:', index)

# string indexing
print(s[0], s[1], s[2], s[3], s[4], s[5])
print(s[-1], s[-2], s[-3], s[-4], s[-5], s[-6])

# string slicing
print(s[:])
print(s[1:4])
print(s[:3])
print(s[3:])
print(s[-3:])
print(s[:3] + s[3:])
print(s[:3] + s[-3:])

# string slicing with offsets
print(s[::1])  # no offset
print(s[::2])  # print every 2nd character in string

# strings are immutable hence assignment throws error
#s[0] = 'X'

# creates a new string
print('Original String id:', id(s))
# creates a new string
s = 'X' + s[1:]
print(s)
print('New String id:', id(s))

# String methods
# case conversions
s = 'python is great'
print(s.capitalize())
print(s.upper())
# converting to title case
print(s.title())

# string replace
print(s.replace('python', 'NLP'))

# Numeric checks
print('12345'.isdecimal())
print('apollo11'.isdecimal())
print('python'.isalpha())
print('number1'.isalpha())
print('total'.isalnum())
print('abc123'.isalnum())
print('1+1'.isalnum())

# string splitting and joining
s = 'I,am,a,comma,separated,string'
print(s.split(','))
print(' '.join(s.split(',')))

# stripping whitespace characters
s = '   I am surrounded by spaces    '
print(s)
print(s.strip())

# some more combinations
sentences = 'Python is great. NLP is also good.'
print(sentences.split('.'))
print('\n'.join(sentences.split('.')))
print('\n'.join([sentence.strip() for sentence in sentences.split('.') if sentence]))

# String formatting
# simple string formatting expressions
print('Hello %s' %('Python!'))
print('Hello %s %s' %('World!', 'How are you?'))

# formatting expressions with different data types
print('We have %d %s containing %.2f gallons of %s' %(2, 'bottles', 2.5, 'milk'))
print('We have %d %s containing %.2f gallons of %s' %(5, 'jugs', 10.867, 'juice'))

# formatting using the format method
print('Hello {} {}, it is a great {} to meet you at {}'.format('Mr.', 'Jones', 'pleasure', 5))
print('Hello {} {}, it is a great {} to meet you at {} o\'clock'.format('Sir', 'Arthur', 'honor', 9))

# alternative ways of using format
print('I have a {food_item} and a {drink_item} with me'.format(drink_item='soda', food_item='sandwich'))
print('The {animal} has the following attributes: {attributes}'.format(animal='dog', attributes=['lazy', 'loyal']))

# Using regular expressions
# importing the re module
import re
# dealing with unicode matching using regexes
s = u'H\u00e8llo'
print(s)

# does not return the special unicode character even if it is alphanumeric
print(re.findall(r'\w+', s))

# need to explicitely specify the unicode flag to detect it using regex
print(re.findall(r'\w+', s, re.UNICODE))

# setting up a pattern we want to use as a regex
# also creating two sample strings
pattern = 'python'
s1 = 'Python is an excellent language'
s2 = 'I love the Python language. I also use Python to build applications at work!'

# match only returns a match if regex match is found at the beginning of the string
print(re.match(pattern, s1))
# pattern is in lower case hence ignore case flag helps
# in matching same pattern with different cases
print(re.match(pattern, s1, flags=re.IGNORECASE))

# printing matched string and its indices in the original string
m = re.match(pattern, s1, flags=re.IGNORECASE)
print('Found match {} ranging from index {} - {} in the string "{}"'.format(m.group(0), m.start(), m.end(), s1))

# match does not work when pattern is not there in the beginning of string s2
print(re.match(pattern, s2, re.IGNORECASE))

# illustrating find and search methods using the re module
print(re.search(pattern, s2, re.IGNORECASE))
print(re.findall(pattern, s2, re.IGNORECASE))

match_objs = re.finditer(pattern, s2, re.IGNORECASE)
print("String:", s2)
for m in match_objs:
    print('Found match "{}" ranging from index {} - {}'.format(m.group(0), m.start(), m.end()))

# illustrating pattern substitution using sub and subn methods
print(re.sub(pattern, 'Java', s2, flags=re.IGNORECASE))
print(re.subn(pattern, 'Java', s2, flags=re.IGNORECASE))

# dealing with unicode matching using regexes
s = u'H\u00e8llo! this is Python \U0001F40D'
print(s)
print(re.findall(r'\w+', s))
print(re.findall(r"[A-Z]\w+", s, re.UNICODE))

emoji_pattern = r"['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(re.findall(emoji_pattern, s, re.UNICODE))



