import string

def remove_extra_spaces(text):
    return ' '.join(text.split())

def space_apart_punctuation(text):
    return ''.join([' ' + char + ' ' if char in string.punctuation else char for char in text])

space_apart = space_apart_punctuation(input_text)
result = remove_extra_spaces(space_apart)
split_data = result.split()

k = 4
rounds = len(split_data) // k + 1

for i in range(0, rounds):
    chunk = ' '.join(split_data[k*i:k*i+k])
    

