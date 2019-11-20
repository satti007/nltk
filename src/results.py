
import os

languages = ['bn', 'gu', 'ml', 'mr', 'ta', 'te']
embeds_srcs = ['indicnlp', 'fastext']

for lang in languages:
    for src in embeds_srcs:
        lang_params = 'python main.py --lang {}'.format(lang)
        if src == 'indicnlp':
            emb_path = (' --emb_path ../data/embeds/{}/indicnlp.v1.{}.vec'.
                        format(src, lang))
            embeds_params = (' --full_vocab True --max_vocab 0 --emb_dim 300')
        else:
            emb_path = (' --emb_path ../data/embeds/{}/cc.{}.300.vec'.
                        format(src, lang))
            embeds_params = (' --full_vocab True --max_vocab {} --emb_dim 300'.
                             format(1000000))

        classify_params = ' --k 4 --valid False'

        cmd = lang_params + emb_path + embeds_params + classify_params
        print(cmd)
        os.system(cmd)
        print('##' * 50)
