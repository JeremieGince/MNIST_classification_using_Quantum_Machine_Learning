def ceasarCipher(message):
    LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for key in range(len(LETTERS)):
        translated = ''

        for symbol in message:
            if symbol in LETTERS:
                num = LETTERS.find(symbol)
                num = num - key
                if num < 0:
                    num = num + len(LETTERS)
                translated = translated + LETTERS[num]
            else:
                translated = translated + symbol
        print('Hacking key #%s: %s' % (key, translated))


def generateKey(string, key):
    key = list(key)
    if len(string) == len(key):
        return(key)
    else:
        for i in range(len(string) - len(key)):
            key.append(key[i % len(key)])
    return "" . join(key)


def vignereCipher(cipherText, key):
    orig_text = []
    for i in range(len(cipherText)):
        x = (ord(cipherText[i]) - ord(key[i]) + 26) % 26
        x += ord('A')
        orig_text.append(chr(x))
    return "".join(orig_text)


if __name__ == '__main__':
    message1 = 'NRUWJXXNAJ BTWP GZY YMJ HFJXFW HNUMJW NX STY XJHZWJ JSTZLM KTW HQFXXNKNJI YWFSXRNXXNTSX'
    message2 = 'YMJ WJFQ RJXXFLJ NX NSHQZIJI GJQTB FSI BFX JSHWDUYJI ZXNSL F ANLSJWJ HNUMJW BNYM PJDBTWI MFHPFYMTS'
    message3 = 'VZZI PGPWAZL AWNQ YXKEFNT NQ ZHL OQHHXGRBWP YLT MCQ BK QSJGDNFBGZ SVI YFQ VGAQLHY RTBFS JCTW'
    encodedMessages = [message1, message2, message3]

    for message in encodedMessages:
        ceasarCipher(message)

    message1 = 'IMPRESSIVE WORK BUT THE CAESAR CIPHER IS NOT SECURE ENOUGH FOR CLASSIFIED TRANSMISSIONS'
    message2 = 'THE REAL MESSAGE IS INCLUDED BELOW AND WAS ENCRYPTED USING A VIGNERE CIPHER WITH KEYWORD HACKATHON'
    message3 = 'QUUD KBKRVUG VRIL TSFZAIO IL UCG JLCCSBMWRK TGO HXL WF LNEBYIAWBU NQD TAL QBVLGCT MOWAN EXOR'.replace(' ', '')
    key = generateKey(message3, 'HACKATHON')

    print(vignereCipher(message3, key))