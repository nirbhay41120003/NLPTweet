import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .predictor import predict_sentiment


@require_http_methods(["GET", "POST"])
def home(request):
    context = {
        "input_text": "",
        "result": None,
        "error": None,
    }

    if request.method == "POST":
        text = request.POST.get("tweet", "").strip()
        context["input_text"] = text

        if not text:
            context["error"] = "Please enter tweet text before running inference."
        else:
            try:
                context["result"] = predict_sentiment(text)
            except Exception as exc:
                context["error"] = f"Model inference failed: {exc}"

    return render(request, "inference/home.html", context)


@require_GET
def health(request):
    return JsonResponse({"status": "ok"})


@require_POST
def predict_api(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Request body must be valid JSON."}, status=400)

    text = str(payload.get("text", "")).strip()
    if not text:
        return JsonResponse({"error": "The 'text' field is required."}, status=400)

    try:
        result = predict_sentiment(text)
    except Exception as exc:
        return JsonResponse({"error": f"Model inference failed: {exc}"}, status=500)

    return JsonResponse({"input": text, "prediction": result})
