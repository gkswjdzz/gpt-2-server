import encoder


def encode(raw_input):
    enc = encoder.get_encoder()
    context_tokens = enc.encode(raw_input)
    return context_tokens


def decode(array):
    enc = encoder.get_encoder()
    text = enc.decode(array[0])

    print(text)
    return text
