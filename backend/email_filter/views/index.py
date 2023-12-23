from django.http import JsonResponse

from email_filter.views import NaiveBayes


def solve(request):
    data = request.GET
    content = data.get("content")
    result = NaiveBayes.main(content)

    return JsonResponse({
        'result': result
    })
