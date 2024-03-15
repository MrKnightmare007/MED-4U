import sqlite3

# Connect to the database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Replace the placeholder image paths with your own paths
update_query = """
    UPDATE products
    SET image = ?
    WHERE productId = ?
"""


# Replace the placeholders with your own image paths and product IDs
new_image_path_15 = 'montina_L.jpg'
product_id_15 = 15
cursor.execute(update_query, (new_image_path_15, product_id_15))

new_image_path_14 = 'ascoril_c.jpg'
product_id_14 = 14
cursor.execute(update_query, (new_image_path_14, product_id_14))

new_image_path_13 = 'pan_40.jpg'
product_id_13 = 13
cursor.execute(update_query, (new_image_path_13, product_id_13))

new_image_path_12 = 'switch_200.jpg'
product_id_12 = 12
cursor.execute(update_query, (new_image_path_12, product_id_12))

new_image_path_11 = 'digene.jpg'
product_id_11 = 11
cursor.execute(update_query, (new_image_path_11, product_id_11))

new_image_path_10 = 'betadine.jpg'
product_id_10 = 10
cursor.execute(update_query, (new_image_path_10, product_id_10))

new_image_path_09 = 'kentakort.jpg'
product_id_09 = 9
cursor.execute(update_query, (new_image_path_09, product_id_09))

new_image_path_08 = 'silverex.jpg'
product_id_08 = 8
cursor.execute(update_query, (new_image_path_08, product_id_08))

new_image_path_07 = 'dart.jpg'
product_id_07 = 7
cursor.execute(update_query, (new_image_path_07, product_id_07))

new_image_path_06 = 'calpol.jpg'
product_id_06 = 6
cursor.execute(update_query, (new_image_path_06, product_id_06))

new_image_path_05 = 'zofer.jpg'
product_id_05 = 5
cursor.execute(update_query, (new_image_path_05, product_id_05))

new_image_path_04 = 'azithral.jpg'
product_id_04 = 4
cursor.execute(update_query, (new_image_path_04, product_id_04))

new_image_path_03 = 'volini.jpg'
product_id_03 = 3
cursor.execute(update_query, (new_image_path_03, product_id_03))

new_image_path_02 = 'burnol.jpg'
product_id_02 = 2
cursor.execute(update_query, (new_image_path_02, product_id_02))

new_image_path_01 = 'ibuprofen.jpg'
product_id_01 = 1
cursor.execute(update_query, (new_image_path_01, product_id_01))

new_image_path_09 = 'drotin.jpg'
product_id_09 = 9
cursor.execute(update_query, (new_image_path_09, product_id_09))
#new_image_path_14 = 'log in.png'
#product_id_14 = 14
#cursor.execute(update_query, (new_image_path_14, product_id_14))
update_query = """
    UPDATE products
    SET name = ?, price = ?
    WHERE productId = ?
"""
# Repeat the above steps for each product
# Update the placeholders with your own values and product IDs
new_name_2 = 'Burnol'
new_price_2 = 150.00
product_id_2 = 2
cursor.execute(update_query, (new_name_2, new_price_2, product_id_2))

new_name_3 = 'Volini'
new_price_3 = 100.00
product_id_3 = 3
cursor.execute(update_query, (new_name_3, new_price_3, product_id_3))

new_name_4 = 'Azithral'
new_price_4 = 400.00
product_id_4 = 4
cursor.execute(update_query, (new_name_4, new_price_4, product_id_4))

new_name_5 = 'Zofer'
new_price_5 = 120.00
product_id_5 = 5
cursor.execute(update_query, (new_name_5, new_price_5, product_id_5))

new_name_6 = 'Calpol'
new_price_6 = 140.00
product_id_6 = 6
cursor.execute(update_query, (new_name_6, new_price_6, product_id_6))

new_name_7 = 'Dart'
new_price_7 = 90.00
product_id_7 = 7
cursor.execute(update_query, (new_name_7, new_price_7, product_id_7))

new_name_8 = 'Silverex Gel'
new_price_8 = 80.00
product_id_8 = 8
cursor.execute(update_query, (new_name_8, new_price_8, product_id_8))
new_name_9 = 'Drotin'
new_price_9 = 130.00
product_id_9 = 9
cursor.execute(update_query, (new_name_9, new_price_9, product_id_9))

new_name_10 = 'Betadine'
new_price_10 = 150.00
product_id_10 = 10
cursor.execute(update_query, (new_name_10, new_price_10, product_id_10))

new_name_11 = 'Digene'
new_price_11 = 75.00
product_id_11 = 11
cursor.execute(update_query, (new_name_11, new_price_11, product_id_11))

new_name_12 = 'Switch 200'
new_price_12 = 175.00
product_id_12 = 12
cursor.execute(update_query, (new_name_12, new_price_12, product_id_12))

new_name_13 = 'Pan 40'
new_price_13 = 70.00
product_id_13 = 13
cursor.execute(update_query, (new_name_13, new_price_13, product_id_13))

new_name_14 = 'Ascoril'
new_price_14 = 180.00
product_id_14 = 14
cursor.execute(update_query, (new_name_14, new_price_14, product_id_14))

new_name_15 = 'Montina L'
new_price_15 = 250.00
product_id_15 = 15

# Update the description for specific products
update_query = """
    UPDATE products
    SET description = ?
    WHERE productId = ?
"""

# Update the placeholders with your own values and product IDs
new_description_2 = 'Burnol is an antiseptic cream commonly used for the treatment of minor burns, scalds, and wounds, providing a soothing and protective layer to promote healing.'
product_id_2 = 2
cursor.execute(update_query, (new_description_2, product_id_2))

new_description_3 = 'Volini is a topical pain relief gel or spray that contains analgesic and anti-inflammatory ingredients, providing quick and targeted relief from muscle and joint pain, sprains, and strains.'
product_id_3 = 3
cursor.execute(update_query, (new_description_3, product_id_3))

new_description_4 = 'Azithral is an antibiotic medication belonging to the macrolide class, commonly used to treat various bacterial infections, including respiratory tract infections, skin infections, and sexually transmitted diseases.'
product_id_4 = 4
cursor.execute(update_query, (new_description_4, product_id_4))

new_description_5 = 'Zofer is an antiemetic medication that belongs to the class of serotonin 5-HT3 receptor antagonists. It is often prescribed to prevent and alleviate nausea and vomiting associated with chemotherapy, radiation therapy, and surgical procedures.'
product_id_5 = 5
cursor.execute(update_query, (new_description_5, product_id_5))

new_description_6 = 'Calpol is a popular brand of paracetamol, a common over-the-counter medication used to alleviate pain and reduce fever. It is widely used for conditions such as headaches, muscle aches, toothaches, and as a general fever reducer, particularly in children.'
product_id_6 = 6
cursor.execute(update_query, (new_description_6, product_id_6))

new_description_7 = 'Dart is often employed for various conditions such as headaches, muscle aches, toothaches, and fever, providing relief for individuals of different age groups.'
product_id_7 = 7
cursor.execute(update_query, (new_description_7, product_id_7))

new_description_8 = 'Silverex is commonly used in the treatment of burns and wound infections to prevent and manage bacterial growth.'
product_id_8 = 8
cursor.execute(update_query, (new_description_8, product_id_8))

new_description_9 = 'Drotin is a brand of drotaverine, a smooth muscle relaxant commonly used to alleviate abdominal pain and discomfort associated with conditions like irritable bowel syndrome (IBS) and menstrual cramps.'
product_id_9 = 9
cursor.execute(update_query, (new_description_9, product_id_9))

new_description_10 = 'Betadine is a brand of povidone-iodine, an antiseptic solution commonly used for wound disinfection and surgical site preparation. It exhibits broad-spectrum antimicrobial properties, effectively killing bacteria, viruses, and fungi. Betadine is widely employed in medical settings for skin disinfection before surgeries, procedures, and for the treatment of minor cuts and abrasions.'
product_id_10 = 10
cursor.execute(update_query, (new_description_10, product_id_10))

new_description_11 = 'Digene is a popular antacid brand that provides relief from acidity, indigestion, and heartburn. It typically contains a combination of aluminum hydroxide, magnesium hydroxide, and simethicone.'
product_id_11 = 11
cursor.execute(update_query, (new_description_11, product_id_11))

new_description_12 = 'Swich 200 Tablet is an antibiotic medicine used to treat bacterial infections in your body. It is effective in infections of the lungs (eg. pneumonia), urinary tract, ear, nasal sinus, throat, and skin.'
product_id_12 = 12
cursor.execute(update_query, (new_description_12, product_id_12))

new_description_13 = 'Pan 40 Tablet is a Tablet manufactured by Aristo Pharmaceuticals. It is commonly used for the diagnosis or treatment of Gastro-esophageal reflux disease, Heartburn, Euophagus inflammation, Stomach ulcers.'
product_id_13 = 13
cursor.execute(update_query, (new_description_13, product_id_13))

new_description_14 = 'Ascoril helps treat chronic bronchitis, in which the airways within the lungs are swollen, reddened, and inflamed.'
product_id_14 = 14
cursor.execute(update_query, (new_description_14, product_id_14))

new_description_15 = 'Montina-L tablet is a medication used to treat allergic rhinitis symptoms such as a runny and stuffy nose, blockage in the airways, sneezing, itching, watery eyes, and other allergic symptoms.'
product_id_15 = 15
cursor.execute(update_query, (new_description_15, product_id_15))

# Commit the changes
conn.commit()

# Close the connection
conn.close()
