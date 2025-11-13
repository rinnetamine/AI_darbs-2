from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from models import Product, CartItem, Order, OrderItem
from database import db
from flask_login import current_user, login_required
from forms import AddToCartForm, CheckoutForm
from chatbot_integration.chatbot_service import ChatbotService

shop_bp = Blueprint('shop', __name__, template_folder='../templates')

# Lazy singleton for chatbot service to avoid raising at import time if env missing
_chatbot_service = None


def get_chatbot_service() -> ChatbotService:
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service

def get_products_from_db():
    """
    Fetches all products from the database and formats them into a simple string for the LLM.
    """
    try:
        products = Product.query.all()
        if not products:
            return "There are currently no products available in the shop."
        
        product_list_str = "Here is a list of available products:\n"
        for p in products:
            product_list_str += f"- Name: {p.name}, Price: ${p.price:.2f}, Stock: {p.stock}\n"
        
        return product_list_str
    except Exception as e:
        print(f"Error fetching products from DB: {e}")
        return "I was unable to access the product catalog."

def get_products_list():
    """
    Retrieve product list from the database in structured format.
    Returns a list of product objects containing all required fields.
    """
    try:
        products = Product.query.all()
        products_list = []
        
        for product in products:
            products_list.append({
                'id': product.id,
                'name': product.name,
                'description': product.description if hasattr(product, 'description') else '',
                'price': float(product.price),
                'stock': product.stock
            })
        
        return products_list
    except Exception as e:
        print(f"Error while fetching products from DB: {e}")
        return []

@shop_bp.route('/chatbot', methods=['POST'])
def chatbot():
    """
    Chatbot endpoint that handles user messages and returns the AI response.

    Expects JSON with:
    - message: user message (string)
    - chat_history: conversation history (list, optional)

    Returns JSON with:
    - response: chatbot reply (string)
    - success: whether the request succeeded (boolean)
    - error: error message if the request failed (string, optional)
    """
    try:
        # Receive data from the request
        data = request.get_json()
        
        # Validate that a message was provided
        if not data or 'message' not in data:
            return jsonify({
                'response': 'Please send a valid message.',
                'success': False,
                'error': 'No message received'
            }), 400
        
        user_message = data.get('message', '').strip()
        chat_history = data.get('chat_history', [])
        
        # Check that the message is not empty
        if not user_message:
            return jsonify({
                'response': 'Please enter a message.',
                'success': False,
                'error': 'Empty message'
            }), 400
        
        # Get structured product list from the database
        products = get_products_list()

        # Validate and limit chat_history to prevent malicious payloads
        if not isinstance(chat_history, list):
            chat_history = []
        # Only accept simple dicts with role and content
        sanitized_history = []
        for item in chat_history[-12:]:
            if isinstance(item, dict) and 'role' in item and 'content' in item:
                sanitized_history.append({'role': str(item['role']), 'content': str(item['content'])})

        # Retrieve (lazy) chatbot service and call it
        chatbot_service = get_chatbot_service()
        response = chatbot_service.get_chatbot_response_with_products(
            user_message=user_message,
            chat_history=sanitized_history,
            products=products
        )
        
        # Return the chatbot response as JSON
        return jsonify(response), 200
        
    except Exception as e:
        # Error handling
        print(f"Error /chatbot endpoint: {str(e)}")
        return jsonify({
            'response': 'Sorry, a server error occurred. Please try again.',
            'success': False,
            'error': str(e)
        }), 500

@shop_bp.route('/shop')
def product_list():
    products = Product.query.all()
    return render_template('shop/product_list.html', title='Shop', products=products)

@shop_bp.route('/product/<int:product_id>', methods=['GET', 'POST'])
def product_detail(product_id):
    product = Product.query.get_or_404(product_id)
    form = AddToCartForm()
    if form.validate_on_submit():
        if not current_user.is_authenticated:
            flash('You need to log in to add items to your cart.', 'info')
            return redirect(url_for('auth.login', next=request.url))

        quantity = form.quantity.data
        if quantity <= 0:
            flash('Quantity must be at least 1.', 'danger')
            return redirect(url_for('shop.product_detail', product_id=product.id))

        if product.stock < quantity:
            flash(f'Insufficient stock. Only {product.stock} left.', 'danger')
            return redirect(url_for('shop.product_detail', product_id=product.id))

        cart_item = CartItem.query.filter_by(user_id=current_user.id, product_id=product.id).first()
        if cart_item:
            cart_item.quantity += quantity
        else:
            cart_item = CartItem(user_id=current_user.id, product_id=product.id, quantity=quantity)
        
        db.session.add(cart_item)
        db.session.commit()
        flash(f'{quantity} x {product.name} added to your cart!', 'success')
        return redirect(url_for('shop.cart'))
    
    return render_template('shop/product_detail.html', title=product.name, product=product, form=form)

@shop_bp.route('/cart')
@login_required
def cart():
    cart_items = current_user.cart_items.all()
    total_price = sum(item.product.price * item.quantity for item in cart_items)
    return render_template('cart.html', title='Your Cart', cart_items=cart_items, total_price=total_price)

@shop_bp.route('/cart/remove/<int:item_id>')
@login_required
def remove_from_cart(item_id):
    cart_item = CartItem.query.get_or_404(item_id)
    if cart_item.user_id != current_user.id:
        flash('You are not authorized to remove this item.', 'danger')
        return redirect(url_for('shop.cart'))
    
    db.session.delete(cart_item)
    db.session.commit()
    flash('Item removed from cart.', 'success')
    return redirect(url_for('shop.cart'))

@shop_bp.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    cart_items = current_user.cart_items.all()
    if not cart_items:
        flash('Your cart is empty!', 'warning')
        return redirect(url_for('shop.product_list'))

    total_amount = sum(item.product.price * item.quantity for item in cart_items)
    checkout_form = CheckoutForm()

    if checkout_form.validate_on_submit():
        new_order = Order(user_id=current_user.id, total_amount=total_amount, status='Processing')
        db.session.add(new_order)
        db.session.flush()

        for item in cart_items:
            order_item = OrderItem(
                order_id=new_order.id,
                product_id=item.product_id,
                quantity=item.quantity,
                price=item.product.price
            )
            product = Product.query.get(item.product_id)
            if product.stock < item.quantity:
                db.session.rollback()
                flash(f'Not enough stock for {product.name}. Please adjust your cart.', 'danger')
                return redirect(url_for('shop.cart'))
            product.stock -= item.quantity
            db.session.add(order_item)
            db.session.delete(item)
        
        db.session.commit()
        flash('Your order has been placed successfully!', 'success')
        return redirect(url_for('shop.purchase_history'))
    
    return render_template('checkout.html', title='Checkout', cart_items=cart_items, total_amount=total_amount, form=checkout_form)

@shop_bp.route('/purchase_history')
@login_required
def purchase_history():
    orders = current_user.orders.order_by(Order.order_date.desc()).all()
    return render_template('purchase_history.html', title='Purchase History', orders=orders)