
class TextAligner:
    def __init__(self, tokenizer, PAD_TOKEN, EPAD_TOKEN):
        self.tokenizer = tokenizer
        self.PAD_TOKEN = PAD_TOKEN
        self.EPAD_TOKEN = EPAD_TOKEN

    def pad(self, frame_list, sequence_length):
        """
        Generate moshi-style text_with_pad with length equals to sequence_length(codec length)
        """
        start_sec = frame_list[0].start_sec
        raw_text = ""
        text = [self.PAD_TOKEN for _ in range(sequence_length)]
        for frame in frame_list:
            timestamp = frame.start_sec - start_sec
            insert_idx = int((timestamp*1000)//80)
            # Seperate two close tokens
            while text[insert_idx] != self.PAD_TOKEN:
                if insert_idx + 1 >= sequence_length:
                    continue  # Skip this text
                else:
                    insert_idx += 1

            text_token = self.tokenizer.encode(frame.content, add_special_tokens=False)
            
            text[insert_idx] = frame.content
            raw_text += frame.content

            # The encoded text can be longer than one token
            for i in range(1, len(text_token)):
                text[insert_idx+i] = "DEL"

        # _text = deepcopy(text)  # For debug
        text = [t for t in text if t != "DEL"]

        # Special case: 兩個連太近的字、中間沒 pad。
        # Example: '以', '前', '以前'都是一個 token，有時 whisper 不會 predict '以前'，而是分成'以'、'前'兩個 frame
        def handler(text_list):
            # 統一 pad 在後面
            # Example: handler(['以', '前', '我', '在']) -> ['以', '前', '我', '在', PAD_TOKEN]
            # 因為「以前我在」會被 tokenize 成 3 個 tokens
            actual_tok_len = len(self.tokenizer.encode("".join(text_list), add_special_tokens=False))
            text_list.extend([self.PAD_TOKEN] * (len(text_list) - actual_tok_len))
            return text_list
        idx = 0
        buffer = []
        new_text = []
        while idx < len(text):
            if text[idx] == self.PAD_TOKEN:
                if buffer:
                    new_text.extend(handler(buffer))
                    buffer = []
                new_text.append(self.PAD_TOKEN)
                idx += 1
                continue
            buffer.append(text[idx])
            idx += 1
        if buffer:
            new_text.extend(handler(buffer))
        text = new_text
        # End special case

        # Add [END_PAD]
        for i in range(len(text)):
            if text[i] != self.PAD_TOKEN and i-1 >= 0 and text[i-1] == self.PAD_TOKEN:
                text[i-1] = self.EPAD_TOKEN

        token_seq = self.tokenizer.encode("".join(text), add_special_tokens=False)
        try:
            assert len(token_seq) == sequence_length, f"{len(token_seq)} != {sequence_length}"

        # Debug info
        except:
            print(f"{len(token_seq)} != {sequence_length}")
            print(" ".join(text))
            # print(_text)
            text = [self.PAD_TOKEN for _ in range(sequence_length)]
            for frame in frame_list:
                timestamp = frame.start_sec - start_sec
                insert_idx = int((timestamp*1000)//80)
                # Seperate two close tokens
                while text[insert_idx] != self.PAD_TOKEN:
                    if insert_idx + 1 >= sequence_length:
                        continue  # Skip this text
                    else:
                        insert_idx += 1

                text_token = self.tokenizer.encode(frame.content, add_special_tokens=False)
                
                print(insert_idx, frame.content, len(text_token))

            exit()
        
        text_with_pad = self.tokenizer.decode(token_seq)
        return raw_text, text_with_pad

