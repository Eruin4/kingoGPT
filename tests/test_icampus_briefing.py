import unittest
from kingogpt.icampus_briefing import context

class BriefingTests(unittest.TestCase):
    def data(self):
        return {'checked_at':'2026-09-20T23:30:00+09:00','status':'collected','gaps':[],
                'courses':[{'id':1,'name':'과목'}], 'coverage':[],
                'items':[{'id':'1:announcement:2','body':'9월 21일까지 제출','url':'https://canvas.skku.edu/courses/1','token':'hidden'}]}
    def test_week_is_calendar_week_in_korea(self):
        result=context(self.data(),[])
        self.assertEqual((result['week_start'],result['week_end']),('2026-09-14','2026-09-20'))
    def test_all_bodies_and_source_links_preserved(self):
        data=self.data();result=context(data,[])
        self.assertEqual(len(result['items']),len(data['items']))
        self.assertEqual(result['items'][0]['body'],data['items'][0]['body'])
        self.assertEqual(result['items'][0]['url'],data['items'][0]['url'])
        self.assertNotIn('token',result['items'][0])
    def test_failed_or_incomplete_results_rejected(self):
        for overrides in ({'status':'partial'},{'gaps':[{'reason':'HTTP 500'}]}):
            with self.assertRaises(ValueError):context({**self.data(),**overrides},[])
