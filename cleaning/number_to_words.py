# Change numbers to words
# Remove numbers from a string using regular expression
import re

num_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', 
            '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

tens_dict = {'10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', 
                '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen', 
                '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty', '60': 'sixty', 
                '70': 'seventy', '80': 'eighty', '90': 'ninety'} 

hundred_dict = {'100': 'one hundred', '200': 'two hundred', '300': 'three hundred', '400': 'four hundred', 
                '500': 'five hundred', '600': 'six hundred', '700': 'seven hundred', '800': 'eight hundred', 
                '900': 'nine hundred'} 

Large_numbers = {100: 'hundred', 1000: 'thousand', 1_000_000: 'million'}

def convert_to_words(num):
    num = int(num)  # Ensure input is an integer

    if num < 10:
        return num_dict[str(num)]
    
    if num < 20:
        return tens_dict[str(num)]
    
    if num < 100:
        tens, remainder = divmod(num, 10)
        return tens_dict[str(tens * 10)] + ('' if remainder == 0 else '-' + num_dict[str(remainder)])
    
    if num < 1000:
        hundreds, remainder = divmod(num, 100)
        return num_dict[str(hundreds)] + ' hundred' + ('' if remainder == 0 else ' ' + convert_to_words(remainder))
    
    if num < 1_000_000:
        thousands, remainder = divmod(num, 1000)
        return convert_to_words(thousands) + ' thousand' + ('' if remainder == 0 else ' ' + convert_to_words(remainder))
    
    for large_num in sorted(Large_numbers.keys(), reverse=True):
        if num >= large_num:
            large, remainder = divmod(num, large_num)
            return convert_to_words(large) + ' ' + Large_numbers[large_num] + ('' if remainder == 0 else ' ' + convert_to_words(remainder))

def convert_numbers_to_words(text):
    if isinstance(text, int):  # If input is a number, convert it directly
        return convert_to_words(text)
    
    if isinstance(text, str):  # If input is text, replace numbers within the string
        text = re.sub(r'\d+', lambda x: convert_to_words(int(x.group())), text)
        return text
    
    return text



# Apply the function
# Testing with a number
input_number = 213423
print(convert_numbers_to_words(input_number))

# Testing with text containing numbers
input_text = "I have 23 oranges and 4354 biscuits! Do you want some?"
print(convert_numbers_to_words(input_text))


