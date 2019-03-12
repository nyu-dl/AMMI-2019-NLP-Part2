SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt
BPE_TOKENS=40000

src=en
tgt=fr
l = $src

cat sample.txt | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> sample_tokenized.txt
            
prep=wmt14_en_fr
BPE_CODE=$prep/code

echo "apply_bpe.py to sample_tokenized.txt..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < sample_tokenized.txt > sample_bpe.txt