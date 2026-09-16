import json
import unittest
from kingogpt.icampus_extract import batches,validate

class ExtractionTests(unittest.TestCase):
    def answer(self,source='x',evidence='제출',date=''):
        return json.dumps({'summary':'요약','actions':[{'source_id':source,'action':'제출하기','evidence':evidence,'date_text':date,'optional':False}]})
    def test_rejects_invented_evidence_and_dates(self):
        rows=[{'id':'x','body':'과제를 제출하세요','title':'과제'}]
        for answer in [self.answer(source='missing'),self.answer(evidence='시험 응시'),self.answer(date='내일')]:
            with self.assertRaises(ValueError):validate(answer,rows)
        self.assertEqual(len(validate(self.answer(),rows)['actions']),1)
    def test_long_body_parts_keep_same_source_evidence(self):
        rows=list(batches([{'id':'x','title':'t','body':'a'*7500}]))[0]
        self.assertEqual(len(validate(self.answer(evidence='a'*1500),rows)['actions']),1)
    def test_batched_source_text_is_preserved(self):
        body='처음'+('x'*16000)+'끝'
        rows=[r for b in batches([{'id':'x','body':body}]) for r in b]
        self.assertTrue(rows[0]['body'].startswith('처음'))
        self.assertTrue(rows[-1]['body'].endswith('끝'))
        rebuilt=rows[0]['body']+''.join(x['body'][500:] for x in rows[1:])
        self.assertEqual(rebuilt,body)
