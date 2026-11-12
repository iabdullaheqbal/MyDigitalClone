# Study notes: machine learning for security

My main research interest is deep learning applied to cybersecurity. These notes summarize the areas I have worked on or explored.

## Intrusion detection systems
An intrusion detection system (IDS) monitors network traffic or host activity and flags malicious behavior. There are two classic approaches. Signature-based detection matches known attack patterns, which is precise but blind to new attacks. Anomaly-based detection learns what normal traffic looks like and flags deviations, which can catch novel attacks but produces more false positives. Machine learning IDS work typically uses flow features such as packet counts, byte counts, durations, flags and ports, and trains classifiers on labeled datasets. The main practical difficulties are class imbalance, because attacks are rare, dataset age and realism, and keeping false positive rates low enough for a human analyst to trust the alerts. I built an AI-powered IDS earlier as a project.

## Phishing detection
Phishing detection can look at the email text, the headers, the URLs and the sender. Text models classify language that creates urgency or asks for credentials. URL features include length, number of subdomains, use of IP addresses, and look-alike domains. I built an email phishing detector with a Chrome extension, so detection could happen in the browser as the user reads a message.

## Adversarial machine learning
Models can be attacked in several ways. Evasion attacks change an input slightly at inference time so the model misclassifies it, a famous example being imperceptible noise on an image. Poisoning attacks corrupt the training data so the model learns something wrong or contains a hidden backdoor. Model extraction and membership inference attacks try to steal the model or learn whether a record was in the training set. Defenses include adversarial training, input validation, and monitoring. This area interests me because it asks how much a model can be trusted when someone is actively trying to fool it.

## Federated learning
In federated learning, many clients train a model on their own local data and only share model updates with a central server, which aggregates them, for example with federated averaging. Raw data never leaves the client, which helps privacy. It is useful in security because organizations often cannot share network logs with each other. Challenges include non-identical data across clients, communication cost, and the fact that updates themselves can still leak information or be poisoned by a malicious client.

## Security of LLM agents
LLM agents use tools, planners and memory. That creates new attack surfaces: prompt injection through retrieved documents or web pages, poisoning of the memory an agent reads later, manipulating the planner so it chooses harmful tool calls, and abusing tool permissions. I explored research directions on multi-agent LLM security, including an intrusion detection proposal for multi-agent systems (MA-LLM-IDS) with Sharina, adversarial attacks on the memory and planner parts of agents, and agentic malware detection frameworks.

## Slopsquatting and supply-chain risk
AI coding assistants sometimes suggest package names that do not exist, a kind of hallucination. Attackers can register those names on registries like PyPI or npm and put malware in them, hoping developers install what the assistant suggested. This is called slopsquatting. In CODEAN, the package screening stage checks candidate packages against OSV.dev vulnerability data and the PyPI and npm registries, and uses a Random Forest classifier over features such as package age, download counts and metadata to flag suspicious ones.

## Digital forensics
For my cyber forensics course I used Autopsy to examine disk images and ExifTool to read metadata. Metadata such as camera model, timestamps and GPS coordinates inside images, documents and media files can reveal when and where a file was created or changed. Forensic work needs a careful workflow: keep originals untouched, work on copies, and produce reports of what was found.
