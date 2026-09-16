import unittest
from kingogpt.icampus_hermes import compact,SYSTEM,finish_answer

class HermesBriefingTests(unittest.TestCase):
    def test_full_source_including_completed_items_reaches_agent_once(self):
        items=[{'id':'done','body':'완료 과제 안내','site_status':'submitted','due_at':None},
               {'id':'todo','body':'이번 주까지 준비할 내용','site_status':'unknown'}]
        result=compact({'items':items,'checked_at':'2026-09-15T08:00:00+09:00'})
        self.assertEqual(len(result['items']),2)
        for old,new in zip(items,result['items']):
            self.assertEqual(old['body'],new['body'])
            self.assertEqual(old['site_status'],new['site_status'])
        self.assertNotIn('source_extractions',result)
        self.assertLess(len(SYSTEM),1000)

    def test_date_and_changes_are_grounded(self):
        payload={'checked_at':'2026-09-16T08:00:00+09:00','week_start':'2026-09-14','week_end':'2026-09-20','changes':[]}
        answer=finish_answer('## 2026-09-16 (화)\n### 오늘 할 일\n없음\n### 주요 변경\n새 공지',payload)
        self.assertIn('2026-09-16 (수)',answer)
        self.assertNotIn('새 공지',answer)
        self.assertIn('변경 없음',answer)
