import torch

class BoardInitializer:
    
    def numbers_to_input_row(num1, num2, row_length, divider_length):
        # Convert numbers to binary strings and reverse them (to start from LSB)
        bin1 = bin(num1)[2:][::-1]
        bin2 = bin(num2)[2:][::-1]

        # Pad the shorter binary string to match the length of the longer one
        max_len = max(len(bin1), len(bin2))
        bin1 = bin1.ljust(max_len, '0')
        bin2 = bin2.ljust(max_len, '0')

        # Create the alternating bit sequence with spaces
        divider = [0] * divider_length
        bit_sequence = divider.copy()
        for b1, b2 in zip(bin1, bin2):
            bit_sequence.extend([int(b1)] + divider + [int(b2)] + divider)

        if(len(bit_sequence) > row_length):
            raise ValueError("row_length is to small for num1 and num2")
        
        # Trim or pad the bit_sequence to match the desired row_length
        bit_sequence = bit_sequence[:row_length]
        bit_sequence += [0] * (row_length - len(bit_sequence))

        # Create a torch tensor from the bit sequence
        tensor = torch.tensor(bit_sequence, dtype=torch.int, device='cpu')

        return tensor
    
    def output_row_to_tensor(row, divider_length):
        i = divider_length
        result = []
        while i < len(row):
            result.append(row[i])
            result += divider_length
        return torch.tensor(result, dtype=torch.int, device='cpu')
    