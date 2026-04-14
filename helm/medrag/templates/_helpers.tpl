{{/*
Common labels
*/}}
{{- define "medrag.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end }}

{{/*
Component selector labels
*/}}
{{- define "medrag.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Namespace — her template içinden erişim
*/}}
{{- define "medrag.namespace" -}}
{{- .Values.global.namespace | default .Release.Namespace -}}
{{- end }}
