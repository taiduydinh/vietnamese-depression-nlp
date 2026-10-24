# External resources

Large third-party resources are intentionally excluded from Git. The reproduction script downloads them when needed.

- Vietnamese Word2Vec: `https://thiaisotajppub.s3-ap-northeast-1.amazonaws.com/publicfiles/wiki.vi.model.bin.gz`
- Vietnamese stopwords: `https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt`
- VnCoreNLP resources are obtained through `py_vncorenlp.download_model(...)`.

For the successful server reproduction, the decompressed Word2Vec binary printed SHA-256:

`1e199c88059aabead223be62281cf4bc8bce5cdb2a74efa0b0d6f3cdfa6942a1`

The stopword file hash used by the reproduction is recorded in `../preprocessing_provenance.json`.

**Important local-archive note:** the uploaded local snapshot contained interrupted/partial copies of `wiki.vi.model.bin` and `wiki.vi.model.bin.gz`; the gzip file fails integrity testing. Those partial copies are deliberately not included here.
