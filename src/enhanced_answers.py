"""
Enhanced Answer Generation Methods
=================================

Verbesserte Methoden zur Erzeugung von vollständigen Sätzen in Antworten
"""

import re
import logging

logger = logging.getLogger("Enhanced-Answers")

def is_complete_sentence(text):
    """
    Prüft, ob ein Text ein vollständiger Satz ist
    
    Args:
        text (str): Der zu prüfende Text
        
    Returns:
        bool: True wenn vollständiger Satz, sonst False
    """
    # Einfache Heuristik für vollständige Sätze
    if not text:
        return False
    
    # Prüfe auf Satzanfang (Großbuchstabe)
    starts_with_capital = text[0].isupper()
    
    # Prüfe auf Satzende (Punkt, Ausrufezeichen, Fragezeichen)
    ends_with_punctuation = text[-1] in '.!?'
    
    # Prüfe minimale Länge und Vorhandensein von Verb (einfache Heuristik)
    words = text.split()
    has_minimum_length = len(words) >= 3
    
    return starts_with_capital and ends_with_punctuation and has_minimum_length

def complete_sentence(text, question=""):
    """
    Ergänzt unvollständige Antworten zu vollständigen Sätzen
    
    Args:
        text (str): Der zu vervollständigende Text
        question (str): Die ursprüngliche Frage
        
    Returns:
        str: Vervollständigter Satz
    """
    if not text or text.isspace():
        return "Keine spezifische Information gefunden."
        
    # Extrahiere Subjekt aus der Frage für bessere Satzergänzung
    question_words = question.split()
    subject = ""
    if question.lower().startswith("was ist"):
        subject = question[7:].strip().rstrip('?')
    elif question.lower().startswith("wie"):
        subject = question.strip().rstrip('?')
    elif question.lower().startswith("warum"):
        subject = question.strip().rstrip('?')
    
    # Wenn Text nicht mit Großbuchstaben beginnt
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    # Wenn Text nicht mit Punkt endet
    if text and text[-1] not in '.!?':
        text += "."
    
    # Wenn Text sehr kurz ist
    if len(text.split()) < 3:
        if subject:
            if "ist" in question.lower():
                text = f"{subject} ist {text}"
            else:
                text = f"Zur Frage nach {subject}: {text}"
        else:
            text = f"Die Antwort lautet: {text}"
    
    return text

def create_comprehensive_answer(question, qa_results):
    """
    Erstellt eine umfassende Antwort mit vollständigen Sätzen
    
    Args:
        question (str): Die gestellte Frage
        qa_results (list): Liste von Antwort-Elementen
        
    Returns:
        str: Formatierte, umfassende Antwort
    """
    if not qa_results:
        return f"Zur Frage '{question}' konnte leider keine relevante Information gefunden werden."
        
    # Hauptantwort aus dem besten Ergebnis
    main_answer = qa_results[0]['text']
    
    # Prüfe, ob die Antwort ein vollständiger Satz ist
    if not is_complete_sentence(main_answer):
        main_answer = complete_sentence(main_answer, question)
    
    # Formatiere die finale Antwort
    final_answer = f"Zu Ihrer Frage '{question}':\n\n"
    final_answer += main_answer + "\n\n"
    
    # Füge ergänzende Informationen hinzu
    if len(qa_results) > 1:
        final_answer += "Ergänzend dazu:\n"
        for result in qa_results[1:3]:  # Maximal 2 zusätzliche Antworten
            answer = result['text']
            if answer and answer != main_answer:
                # Stelle sicher, dass auch diese Antworten vollständige Sätze sind
                if not is_complete_sentence(answer):
                    answer = complete_sentence(answer, question)
                final_answer += f"- {answer}\n"
    
    return final_answer

def generate_enhanced_extractive_answer(question, relevant_chunks, relevant_texts, sources):
    """
    Generiert eine verbesserte extraktive Antwort mit vollständigen Sätzen
    
    Args:
        question (str): Die Frage
        relevant_chunks (list): Die relevantesten Chunks
        relevant_texts (list): Extrahierte relevante Textstellen
        sources (list): Quellen der Antworten
        
    Returns:
        str: Extrahierte und verbesserte Antwort
        list: Verwendete Quellen
    """
    # Generiere Antwort basierend auf relevanten Texten
    if relevant_texts:
        # Einleitung mit vollständigem Satz
        answer = f"Hier sind die relevanten Informationen zu Ihrer Frage '{question}':\n\n"
        
        # Füge vollständige Sätze hinzu
        for i, text in enumerate(relevant_texts[:5]):  # Top 5 relevante Texte
            # Stelle sicher, dass jede Antwort ein vollständiger Satz ist
            if not is_complete_sentence(text):
                text = complete_sentence(text, question)
            answer += f"{i+1}. {text}\n\n"
            
        # Abschließender Satz
        answer += "Ich hoffe, diese Informationen helfen Ihnen weiter."
    else:
        # Generische Antwort mit vollständigen Sätzen
        answer = f"Entschuldigung, ich konnte in den vorliegenden Dokumenten keine spezifischen Informationen zu Ihrer Frage '{question}' finden. Können Sie Ihre Frage umformulieren oder ein anderes Thema anfragen?"
    
    return answer, sources
