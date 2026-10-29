# Sampling temperature examples

All runs use the same first 100 frozen validation IDs and seed 42. Changed-output counts compare exact scored strings with greedy decoding; they are not quality measures.

## transformer

- t0.3: 100/100 outputs differ from greedy
- t0.7: 100/100 outputs differ from greedy

### validation_0000000

Source: Students of the Dattatreya city Municipal corporation secondary school demonstrated their imagination power by creating the fictitious fort "Duttgarh".

Reference: महानगर पालिका अंतर्गत दत्तात्रय नगर माध्यमिक स्कूल के विद्यार्थियों ने काल्पनिक किला 'दत्तगढ़' बनाकर अपनी कल्पनाशक्ति का परिचय दिया।

- greedy: राष्ट्रपति ने कहा कि इस प्रकार के लिए, राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन : राष्ट्रपति भवन :
- t0.3: राष्ट्रपति ने कहा कि इस क्षेत्र में ‘स्‍त्याणिक शिक्षा के लिए, राष्ट्रपति भवन :
- t0.7: 19482198 ‘422 it को संबोधित किया गया है, राष्ट्रपति भवन : राष्ट्रपति भवन : 21.

### validation_0000001

Source: With encouragement from Principal Sandhya Medpallivaar the teachers and students built the fort out of clay.

Reference: प्रधानाध्यापक संध्या मेडपल्लीवार के प्रोत्साहित करने पर शिक्षकों व विद्यार्थियों ने मिट्टïी से किले का निर्माण किया।

- greedy: एक शास्‍त्याध्याध्याणिक एवं मंडल के लिए एकत्रिकी के लिए एकत्रिकी के लिए एकत्रिकी का एकत्रिकी।
- t0.3: ढे में फोल्सन के सार के लिए एकत्रिकी और स्फ्छुपे के लिए एक अह्यावल के लिए एक क्रेडल के लिए एकल का एक और निहित है।
- t0.7: अन्य कल्याण के लिए, प्रदायवलोर नेड़दू के तहत सेवा करने के लिए क्षमताओं की निगोन कर दिया।

## mt5

- t0.3: 99/100 outputs differ from greedy
- t0.7: 100/100 outputs differ from greedy

### validation_0000000

Source: Students of the Dattatreya city Municipal corporation secondary school demonstrated their imagination power by creating the fictitious fort "Duttgarh".

Reference: महानगर पालिका अंतर्गत दत्तात्रय नगर माध्यमिक स्कूल के विद्यार्थियों ने काल्पनिक किला 'दत्तगढ़' बनाकर अपनी कल्पनाशक्ति का परिचय दिया।

- greedy: उनके राष्ट्रीय राष्ट्रीय राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय विद्यालय और राष्ट्रीय
- t0.3: मुम्बिया और राष्ट्रीय राष्ट्रीय विद्यालय में तीव्र विकास के लिए राष्ट्रीय राष्ट्रीय और राष्ट्रीय विकास और राष्ट्रीय राष्ट्रीय विद्यालय में स्थापित करने और उन्हें और राष्ट्रीय विकास के लिए उन्हें द्वारा स्थापित करने के लिए किया गया।
- t0.7: उद्योगों ने अपने उच्च विद्यालय विद्यालय के विद्यालय में देखते हैं, लेकिन परिषद ग्रन्थों के स्थान के तीसरे ही दृश्यीकरण की गई अंतिम थीं।

### validation_0000001

Source: With encouragement from Principal Sandhya Medpallivaar the teachers and students built the fort out of clay.

Reference: प्रधानाध्यापक संध्या मेडपल्लीवार के प्रोत्साहित करने पर शिक्षकों व विद्यार्थियों ने मिट्टïी से किले का निर्माण किया।

- greedy: उनके लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए लिए और शिक्षा के लिए
- t0.3: मुरना
- t0.7: ब्यापी

## llama

- t0.3: 82/100 outputs differ from greedy
- t0.7: 93/100 outputs differ from greedy

### validation_0000000

Source: Students of the Dattatreya city Municipal corporation secondary school demonstrated their imagination power by creating the fictitious fort "Duttgarh".

Reference: महानगर पालिका अंतर्गत दत्तात्रय नगर माध्यमिक स्कूल के विद्यार्थियों ने काल्पनिक किला 'दत्तगढ़' बनाकर अपनी कल्पनाशक्ति का परिचय दिया।

- greedy: Here is the translation of the English text into Hindi:  शिक्षकों के द्वारा दत्तात्रेय शहर में मunicipal कार्यालय सेकंड्री स्कूल में प्रदर्शन किया गया जब उन्होंने "दुत्तगार" के नाम से एक झूठी किला बनाई।
- t0.3: मunicipal कार्यालय दत्तात्रेय शहर के सेकेंडरी स्कूल ने अपने विचार शक्ति को प्रदर्शित करने के लिए "दुत्तगार" नाम का सीमित सैन्य किला बनाया।
- t0.7: मunicipal कार्रवाई दत्तात्रेय शहर सेकेंडरी स्कूल मूल्यों के छात्रों ने अपने लिए कल्पना क्षमता को प्रदर्शित करने के लिए "डुट्टगार" की कल्पनात्मक किला बनाई।

### validation_0000001

Source: With encouragement from Principal Sandhya Medpallivaar the teachers and students built the fort out of clay.

Reference: प्रधानाध्यापक संध्या मेडपल्लीवार के प्रोत्साहित करने पर शिक्षकों व विद्यार्थियों ने मिट्टïी से किले का निर्माण किया।

- greedy: The translation of the English text into Hindi is:  स्कूल प्रिंसिपल संदhya मेदपालीवार के प्रोत्साहन के साथ शिक्षकों और छात्रों ने क्लाय के साथ मिलकर किले का निर्माण किया।
- t0.3: बांध के निर्माण के लिए प्रिंसिपल संदhya मध्यपालीवार की सहायता से शिक्षक और छात्र ने कार्य में भाग लिया।
- t0.7: निर्देश (Nirdesh) sandhya medpallivaar

## gemma

- t0.3: 69/100 outputs differ from greedy
- t0.7: 95/100 outputs differ from greedy

### validation_0000000

Source: Students of the Dattatreya city Municipal corporation secondary school demonstrated their imagination power by creating the fictitious fort "Duttgarh".

Reference: महानगर पालिका अंतर्गत दत्तात्रय नगर माध्यमिक स्कूल के विद्यार्थियों ने काल्पनिक किला 'दत्तगढ़' बनाकर अपनी कल्पनाशक्ति का परिचय दिया।

- greedy: शहर के दत्तप्रतीर्थ नगरपालिका माध्यमिक विद्यालय के छात्रக்கள் তাদের কল্পনার शक्ति का प्रदर्शन करते हुए "डुटगढ़" नामक भ्रामक किले की रचना करने में सफल रहे।
- t0.3: शहर के दत्तत्रय नगरपालिका माध्यमिक विद्यालय के छात्र अपनी कल्पना शक्ति का प्रदर्शन करते हुए, "डुटगढ़" नामक भ्रामक किले की रचना करने में सफल रहे।
- t0.7: शहर के दतत्रय नगरपालिका माध्यमिक विद्यालय के छात्रக்கள் своего কল্পনার शक्ति का प्रदर्शन करते हुए, उन्होंने "डल्टिगर्म" नामक भ्रामक किले की रचना की।

### validation_0000001

Source: With encouragement from Principal Sandhya Medpallivaar the teachers and students built the fort out of clay.

Reference: प्रधानाध्यापक संध्या मेडपल्लीवार के प्रोत्साहित करने पर शिक्षकों व विद्यार्थियों ने मिट्टïी से किले का निर्माण किया।

- greedy: शिक्षक सैंडhya मेडपैलिवारा की प्रोत्साहन से, शिक्षक और छात्र मिट्टी से एक किला बनाने लगे।
- t0.3: शिक्षक सैंडhya मेडपैलिवारा की encouragement से, शिक्षक और छात्र मिट्टी से एक किला बनाने लगे।
- t0.7: शिक्षक सैंडhya मेडपैलिवारा के प्रोत्साहन से, शिक्षक और छात्र उन्होंने मिट्टी से एकचाट का किला बनाया।
