from os.path import join

from datasets import load_dataset

dataset = load_dataset('race', 'high')
texts = dataset['train']['article']
print(len(texts))

save_folder = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/Race/'
for i in range(100):
    print(i, texts[i])
    # text = texts[i * 4]
    # with open(join(save_folder, f'{i}.txt'), 'w') as f:
    #     f.write(text)
print("Done")

"""
Статьи повторяются.
Сложные  статьи:
1. Список. (много пунктов, дат и цифр)
2. История. (Много имен названий и дат)
3. Диалоги. (Имена, местоимения, восклицания и вопросы. Мало смысла)

"""